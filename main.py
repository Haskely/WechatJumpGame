import threading
from time import sleep
import scrcpy
import cv2
import numpy as np

def matchTemplate(ori_img, template):
    res = cv2.matchTemplate(ori_img, template, cv2.TM_SQDIFF_NORMED,
                            None, (template != 0).all(-1).astype('uint8'))
    confidence = 1 - res.min()/res.mean()
    top_left = np.unravel_index(res.argmin(), res.shape)[::-1]
    return confidence, top_left


checker_templ = cv2.imread('checker.png')


def get_checker_pos(frame):
    confidence, top_left = matchTemplate(frame, checker_templ)
    if (confidence > 0.87):
        h, w = checker_templ.shape[:2]
        checker_pos = (top_left[0] + w/2, top_left[1] + h - w*0.577/2)
        return checker_pos
    else:
        print(f"没有探测到小跳棋! 因为信心度为{confidence:g} <= 0.87")
        return None


button_templ = cv2.imread('button.png')


def get_button_pos(frame):
    confidence, top_left = matchTemplate(frame, button_templ)
    if (confidence > 0.95):
        h, w = button_templ.shape[:2]
        button_pos = (top_left[0] + w/2, top_left[1] + h/2)
        return button_pos
    else:
        print(f"没有探测到按钮! 因为信心度为{confidence:g} <= 0.95")
        return None


def get_boxcenter_poses(frame: np.ndarray, checker_pos: tuple):
    top_y = None
    target_loc = None

    # 避免把棋子顶端当作方块顶端
    if checker_pos[0] < frame.shape[1] / 2:  # 如果棋子在屏幕左边，目标方块一定在棋子右边
        b = round(checker_pos[0] + checker_templ.shape[1] / 2)
        e = frame.shape[1]
    else:  # 如果棋子在屏幕右边，目标方块一定在棋子左边
        b = 0
        e = round(checker_pos[0] - checker_templ.shape[1] / 2)

    row_start = 200
    c_sen = 25
    
    for i in range(row_start, frame.shape[0]):
        h = frame[i, b:e]
        xs, ys = np.where((h - h[0])**2 > c_sen)
        if xs.shape[0]:
            top_y = i
            x = np.mean(xs) + b
            det_y = 0.577 * abs(x - cen_loc[0]) - \
                abs(top_y - cen_loc[1])  # 利用绝对中心找到偏移量
            y = top_y + abs(det_y)
            target_loc = (x, y)
            break

    # 计算上一次的跳跃目标
    source_loc = (2 * cen_loc[0] - target_loc[0],
                  2 * cen_loc[1] - target_loc[1])

    return target_loc, source_loc


def distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def press(pos, sec):
    phone.control.touch(x=pos[0], y=pos[1], action=scrcpy.ACTION_DOWN)
    sleep(sec)
    phone.control.touch(x=pos[0], y=pos[1], action=scrcpy.ACTION_UP)
    # threading.Timer(sec, phone.control.touch,kwargs=dict(x=pos[0],y=pos[1],action = scrcpy.ACTION_UP))


errorN = 0
k = 4.0e-3
# lr = 1e-6
thinkN = 0
sameN = 0
pre_frame = None
ready2think = False

diff = 0

checker_pos = None
button_pos = None
tar_box_pos = None
src_box_pos = None


def think(frame):
    if frame is None:
        return
    global diff, thinkN, sameN, pre_frame, ready2think, errorN, checker_pos, button_pos, tar_box_pos, src_box_pos

    thinkN = (thinkN+1) % 100
    if not ready2think:
        if sameN < 5:
            diff = (frame != pre_frame).sum() / \
                frame.size if pre_frame is not None else float('inf')
            if diff < 0.05: # 95%像素一致
                sameN += 1
        else:
            # 画面已经静止
            ready2think = True
            sameN = 0

    # cv2.imshow('frame',frame)
    # cv2.waitKey(int(1000 / phone.max_fps))
    # cv2.imshow('diff',frame - pre_frame)
    # cv2.waitKey(int(1000 / phone.max_fps))
        sleep(0.1)
    else:
        ready2think = False
        checker_pos = get_checker_pos(frame)
        if checker_pos:
            # 计算距离并跳跃
            tar_box_pos, src_box_pos = get_boxcenter_poses(frame, checker_pos)

            err_dis = distance(checker_pos, src_box_pos) * \
                (-1.0 if checker_pos[1] < src_box_pos[1] else 1.0)
            # k = k + err_dis*lr
            # print(f'err_dis={err_dis:.2g}\nlr={lr:g}\n{k:g}=k+{err_dis*lr:g}\n')

            dis = distance(checker_pos, tar_box_pos)
            sec = dis*k
            print(f'err_dis={err_dis:g}\nk={k:g}\ndis={dis:g}\nsec={sec:g}\n')
            press((frame.shape[1]/2, frame.shape[0]/2), sec)

        else:
            button_pos = get_button_pos(frame)
            if button_pos:
                # 点击按钮开始游戏
                press(button_pos, 0.2)
            else:
                errorN += 1
                print(f"既没有检测到小跳棋也没有检测到开始游戏按钮 x {errorN}")
                if errorN >= 3:
                    print("程序结束，按任意键退出~")
                    input()
                    stop()

        sleep(0.5)
        checker_pos, button_pos, tar_box_pos, src_box_pos = None, None, None, None

    pre_frame = frame

pre_thinkN = 0

def show_frame(frame):
    if frame is None:
        return

    frame2show = frame.copy()

    def putText(text, pos):
        texts = text.split('\n')
        for i, t in enumerate(texts):
            cv2.putText(frame2show, t, (pos[0], pos[1]+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    putText(
        f'diff = {diff:g}\nthinkN = {thinkN}\nsameN = {sameN}\nready2think = {ready2think}', (5, 50))

    def int_pos(pos):
        return round(pos[0]), round(pos[1])
    
    cv2.circle(frame2show, int_pos(cen_loc),
                2, (255,255, 0), -1)  # 中心点
        
    if checker_pos is not None:
        cv2.circle(frame2show, int_pos(checker_pos),
                   5, (100, 0, 255), -1)  # 棋子的落脚点

    if button_pos is not None:
        h, w = button_templ.shape[:2]
        top_left = (button_pos[0] - w/2, button_pos[1] - h/2)
        bottom_right = (button_pos[0] + w/2, button_pos[1] + h/2)
        cv2.rectangle(frame2show, int_pos(top_left),
                      int_pos(bottom_right), 255, 2)

    if tar_box_pos is not None:
        cv2.circle(frame2show, int_pos(tar_box_pos),
                   5, (0,0,0), -1)  # 目标方块的中心点
        if checker_pos is not None:
            cv2.line(frame2show, int_pos(checker_pos), int_pos(
                tar_box_pos), (0, 255, 255), 2)  # 目标方块中心点 与 当前方块中心点 连线

            def midpoint(p1, p2):
                return (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            line_mid = midpoint(checker_pos, tar_box_pos)
            dis = distance(checker_pos, tar_box_pos)
            sec = dis*k
            putText(f'dis={dis:.3g}\nsec={sec:.3g} s\nk={k:.3g}',
                    int_pos((line_mid[0], line_mid[1] + 10)))

            global pre_thinkN
            if pre_thinkN != thinkN:
                cv2.imwrite('last_info_sc.png', frame2show)
                pre_thinkN = thinkN

    if src_box_pos is not None:
        cv2.circle(frame2show, int_pos(src_box_pos),
                   2, (0, 255, 255), -1)  # 当前方块的中心点

    cv2.imshow('viz', frame2show)
    cv2.waitKey(int(1000 / phone.max_fps))


phone = scrcpy.Client(
    max_width=800,
    bitrate=8000000,
    max_fps=10,
    # stay_awake=True,
    # lock_screen_orientation=False,
)

cen_loc = None
def on_init():
    global cen_loc
    print(
        f"拿到手机!\n\tDevice Name:{phone.device_name}\n\tResolution:{phone.resolution}")
    cen_loc = (phone.resolution[0]/2, phone.resolution[1]/2 + 5)

phone.add_listener(scrcpy.EVENT_INIT, on_init)
# def on_frame(frame):
#     pass
# phone.add_listener(scrcpy.EVENT_FRAME,on_frame)

phone.start(threaded=True)

running = True


def run():
    # 新设想，只调用phone.last_frame,可以避免多线程
    while running:
        think(phone.last_frame)


def show():
    while running:
        show_frame(phone.last_frame)


threading.Thread(target=run).start()
threading.Thread(target=show).start()


def stop():
    global running
    phone.stop()
    running = False

while running:
    if input('输入 end 停止\n') == 'end':
        stop()
