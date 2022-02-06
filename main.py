import numpy as np
import cv2
import scrcpy
from time import sleep
import threading
import ctypes
# Set DPI Awareness  (Windows 10 and 8) https://stackoverflow.com/questions/44398075/can-dpi-scaling-be-enabled-disabled-programmatically-on-a-per-session-basis
errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(0)

POS_T = tuple[float, float]

TAN = 0.5771 # 透视角度 tan

def matchTemplate(ori_img: np.ndarray, template: np.ndarray) -> tuple[float, POS_T]:
    """模板匹配方法进行目标位置检测

    Args:
        ori_img (np.ndarray): 整张图片，高x宽x3(bgr)
        template (np.ndarray): 模板，高x宽x3(bgr)，其中bgr全为0的点默认为忽略mask

    Returns:
        tuple[float,pos_T]: 可信度与匹配目标的左上角坐标
    """
    res = cv2.matchTemplate(ori_img, template, cv2.TM_SQDIFF_NORMED,
                            None, (template != 0).all(-1).astype('uint8'))
    confidence = 1 - res.min()/res.mean()
    top_left = np.unravel_index(res.argmin(), res.shape)[::-1]
    return confidence, top_left


checker_templ = cv2.imread('checker.png')  # 棋子匹配模板图像
def get_checker_pos(frame: np.ndarray) -> POS_T:
    """获取小棋子的落脚点

    Args:
        frame (np.ndarray): 某一帧画面,高x宽x3(bgr)

    Returns:
        pos_T: 小棋子的落脚点坐标(x,y)，x横向，y纵向；若没有小棋子则返回None
    """
    confidence, top_left = matchTemplate(frame, checker_templ)
    if (confidence > 0.87):
        h, w = checker_templ.shape[:2]
        checker_pos = (top_left[0] + w/2, top_left[1] + h - w*TAN/2)
        return checker_pos
    else:
        # print(f"没有探测到小跳棋! 因为信心度为{confidence:g} <= 0.87")
        return None


button_templ = cv2.imread('button.png')  # 按钮匹配模板图像
def get_button_pos(frame: np.ndarray) -> POS_T:
    """获取开始游戏按钮或重新开始按钮的中间坐标

    Args:
        frame (np.ndarray): 某一帧画面,高x宽x3(bgr)

    Returns:
        pos_T: 开始游戏按钮或重新开始按钮的中间坐标(x,y)，x横向，y纵向；若没有按钮则返回None
    """
    confidence, top_left = matchTemplate(frame, button_templ)
    if (confidence > 0.95):
        h, w = button_templ.shape[:2]
        button_pos = (top_left[0] + w/2, top_left[1] + h/2)
        return button_pos
    else:
        # print(f"没有探测到按钮! 因为信心度为{confidence:g} <= 0.95")
        return None


target_top_line = None
def get_platcenter_poses(frame: np.ndarray, checker_pos: POS_T) -> tuple[POS_T, POS_T]:
    """获取游戏中落脚点的中心坐标

    Args:
        frame (np.ndarray): 某一帧画面,高x宽x3(bgr)
        checker_pos (pos_T): 当前帧小棋子的落脚点

    Returns:
        tuple[pos_T,pos_T]: 目标落脚点的中心坐标和当前落脚点的中心坐标
    """
    top_y = None
    target_loc = None

    # 避免把棋子顶端当作平台顶端
    if checker_pos[0] < frame.shape[1] / 2:  # 如果棋子在屏幕左边，目标平台一定在棋子右边
        b = round(checker_pos[0] + checker_templ.shape[1] / 2 + 10)
        e = frame.shape[1] - 20
    else:  # 如果棋子在屏幕右边，目标平台一定在棋子左边
        b = 20
        e = round(checker_pos[0] - checker_templ.shape[1] / 2)

    row_start = 200
    c_sen = 150
    global target_top_line
    for i in range(row_start, frame.shape[0]):
        h = frame[i, b:e]
        xs = np.where(((h - h[0])**2).sum(-1) > c_sen)[0]
        if xs.shape[0]:
            top_y = i
            x = np.mean(xs) + b
            det_y = TAN * abs(x - cen_loc[0]) - \
                abs(top_y - cen_loc[1])  # 利用绝对中心找到偏移量
            y = top_y + abs(det_y)
            target_loc = (x, y)
            target_top_line = ((xs[0] + b, top_y), (xs[-1] + b, top_y))
            break

    # 计算上一次的跳跃目标
    source_loc = (2 * cen_loc[0] - target_loc[0],
                  2 * cen_loc[1] - target_loc[1])

    return target_loc, source_loc


def distance(pos1: POS_T, pos2: POS_T) -> float:
    """计算欧氏距离

    Args:
        pos1 (pos_T): 坐标1
        pos2 (pos_T): 坐标2

    Returns:
        float: 两个坐标的欧氏距离
    """
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def press(pos: POS_T, sec: float) -> None:
    """长按屏幕

    Args:
        pos (pos_T): 按压位置
        sec (float): 按压时间，单位秒
    """
    phone.control.touch(x=pos[0], y=pos[1], action=scrcpy.ACTION_DOWN)
    sleep(sec)
    phone.control.touch(x=pos[0], y=pos[1], action=scrcpy.ACTION_UP)
    # threading.Timer(sec, phone.control.touch,kwargs=dict(x=pos[0],y=pos[1],action = scrcpy.ACTION_UP))


errorN = 0  # 既没有检测到小跳棋也没有检测到开始游戏按钮的次数，超过三次停止程序
k = 4.05e-3  # 比例系数 = 按压时间 / 欧氏距离

thinkN = 0  # 电脑玩家思考的次数 % 1e7

checker_pos = None
button_pos = None
tar_plat_pos = None
src_plat_pos = None

sameN = 0  # 判断当前画面是否已经静止
diff = 0  # 判断当前画面是否已经静止
pre_checker_pos = None  # 判断当前画面是否已经静止
def ready2action(checker_pos):
    global sameN, pre_checker_pos, diff
    if sameN >= show_FPS // 2:
        sameN = 0
        pre_checker_pos = None
        res = True
    else:
        res = False
        if pre_checker_pos is not None:
            diff = distance(pre_checker_pos, checker_pos)
            if diff < 1e-5:
                sameN += 1
        pre_checker_pos = checker_pos
    return res


last_action_frame = None
def think(frame: np.ndarray) -> None:
    """电脑玩家处理一帧画面

    Args:
        frame (np.ndarray): 某一帧画面,高x宽x3(bgr)
    """
    if frame is None:
        return
    global thinkN, errorN, checker_pos, button_pos, tar_plat_pos, src_plat_pos

    thinkN += 1
    checker_pos = get_checker_pos(frame)
    if checker_pos:
        tar_plat_pos, src_plat_pos = get_platcenter_poses(
            frame, checker_pos)
        if ready2action(checker_pos):
            # 计算距离并跳跃

            press((frame.shape[1]/2, frame.shape[0]/2),
                  distance(checker_pos, tar_plat_pos)*k)

            checker_pos, tar_plat_pos, src_plat_pos = None, None, None

            global last_action_frame
            last_action_frame = frame
    else:
        button_pos = get_button_pos(frame)
        if button_pos:
            # 点击按钮开始或重新开始游戏
            press(button_pos, 0.2)
            button_pos = None

            if last_action_frame is not None:
                cv2.imwrite('last_action_frame.png', last_action_frame)


lastsc_thinkN = 0
showN = 0
def show_frame(frame: np.ndarray) -> None:
    """绘制必要信息并展示一帧画面

    Args:
        frame (np.ndarray): 某一帧画面,高x宽x3(bgr)
    """
    if frame is None:
        return
    global showN
    showN += 1
    frame2show = frame.copy()

    def putText(text, pos, color=(71, 99, 255)):
        texts = text.split('\n')
        for i, t in enumerate(texts):
            cv2.putText(frame2show, t, (pos[0], pos[1]+i*15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    info = {
        'frame_FPS': frame_FPS,
        'think_FPS': think_FPS,
        'show_FPS': show_FPS,
        'frameN': frameN,
        'thinkN': thinkN,
        'showN': showN,
        'diff': diff,
        'sameN': sameN,
    }
    putText('\n'.join([f'{key}={val:g}' for key,
            val in info.items()]), (10, 20), color=(255, 191, 0))

    def int_pos(pos):
        return round(pos[0]), round(pos[1])

    cv2.circle(frame2show, int_pos(cen_loc),
               2, (255, 255, 0), -1)  # 中心点

    if checker_pos is not None:
        cv2.circle(frame2show, int_pos(checker_pos),
                   3, (100, 0, 255), -1)  # 棋子的落脚点

    if button_pos is not None:
        h, w = button_templ.shape[:2]
        top_left = (button_pos[0] - w/2, button_pos[1] - h/2)
        bottom_right = (button_pos[0] + w/2, button_pos[1] + h/2)
        cv2.rectangle(frame2show, int_pos(top_left),
                      int_pos(bottom_right), 255, 2)
    if pre_checker_pos is not None:
        cv2.circle(frame2show, int_pos(pre_checker_pos),
                   5, (0, 100, 200), 1)  # 棋子上一帧的落脚点
    if target_top_line is not None:
        cv2.line(frame2show, int_pos(target_top_line[0]), int_pos(
            target_top_line[1]), (0, 0, 200), 2)  # 平台扫描起始线
    if tar_plat_pos is not None:
        cv2.circle(frame2show, int_pos(tar_plat_pos),
                   5, (0, 0, 0), -1)  # 目标平台的中心点
        if checker_pos is not None:
            cv2.line(frame2show, int_pos(checker_pos), int_pos(
                tar_plat_pos), (0, 255, 255), 2)  # 目标平台中心点 与 当前平台中心点 连线

            def midpoint(p1, p2):
                return (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            line_mid = midpoint(checker_pos, tar_plat_pos)
            dis = distance(checker_pos, tar_plat_pos)
            sec = dis*k
            putText(f'dis={dis:.3g}\nsec={sec:.3g} s\nk={k:.3g}',
                    int_pos((line_mid[0], line_mid[1] + 10)))

            global lastsc_thinkN
            if lastsc_thinkN != thinkN:
                cv2.imwrite('last_info_sc.png', frame2show)
                lastsc_thinkN = thinkN

    if src_plat_pos is not None:
        cv2.circle(frame2show, int_pos(src_plat_pos),
                   2, (0, 255, 255), -1)  # 当前平台的中心点

    cv2.imshow(u'WechatJumpGame', frame2show)
    cv2.waitKey(int(1000 / phone.max_fps))


phone = scrcpy.Client(
    max_width=800,
    bitrate=8000000,
    max_fps=60,
)

cen_loc = None
def on_init():
    global cen_loc
    print(
        f"拿到手机!\n\tDevice Name:{phone.device_name}\n\tResolution:{phone.resolution}")
    cen_loc = (phone.resolution[0]/2, phone.resolution[1]/2 + 5)


phone.add_listener(scrcpy.EVENT_INIT, on_init)

FPS_lock = threading.Lock()
frameN = 0
def on_frame(frame):
    global frameN
    with FPS_lock:
        frameN += 1


phone.add_listener(scrcpy.EVENT_FRAME, on_frame)

phone.start(threaded=True)

playing = True


def play():
    while playing:
        think(phone.last_frame)


def show():
    while playing:
        show_frame(phone.last_frame)


frame_FPS = 0
think_FPS = 0
show_FPS = 0
def cal_fps():
    global frame_FPS, frameN, think_FPS, thinkN, show_FPS, showN
    while playing:
        with FPS_lock:
            frameN = 0
            thinkN = 0
            showN = 0
        sleep(1.0)
        frame_FPS = frameN
        think_FPS = thinkN
        show_FPS = showN


threading.Thread(target=play).start()
threading.Thread(target=show).start()

threading.Thread(target=cal_fps).start()


def stop():
    global playing
    phone.stop()
    playing = False


while playing:
    if input('输入 end 停止\n') == 'end':
        stop()
