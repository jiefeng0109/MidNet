import math
import os

import numpy as np
from progress.bar import Bar
from scipy.optimize import fmin, fminbound


def dis(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def cal_k(obj):
    x0, y0 = obj[0][0], obj[0][1]
    x1, y1, c1 = obj[1][0], obj[1][1], obj[1][6]
    x2, y2, c2 = obj[2][0], obj[2][1], obj[2][6]
    x3, y3, c3 = obj[3][0], obj[3][1], obj[3][6]
    x4, y4, c4 = obj[4][0], obj[4][1], obj[4][6]

    sum_con = sum([obj[1][6], obj[2][6], obj[3][6], obj[4][6]])

    def f(k):
        d1 = (k * x1 - y1 - k * x0 + y0) ** 2 / (k ** 2 + 1)
        d2 = (k * x3 - y3 - k * x0 + y0) ** 2 / (k ** 2 + 1)
        d3 = (-1 / k * x2 - y2 + 1 / k * x0 + y0) ** 2 / (1 / k ** 2 + 1)
        d4 = (-1 / k * x4 - y4 + 1 / k * x0 + y0) ** 2 / (1 / k ** 2 + 1)
        return d1 * c1 / sum_con + d2 * c2 / sum_con + d3 * c3 / sum_con + d4 * c4 / sum_con

    min_global = fminbound(f, -100, 100)

    return min_global


def GeneralEquation(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def GetIntersectPointofLines(x1=0, y1=0, x2=0, y2=0, x3=0, y3=0, x4=0, y4=0,
                             A1=None, B1=None, C1=None, A2=None, B2=None, C2=None):
    if A1 == None:
        A1, B1, C1 = GeneralEquation(x1, y1, x2, y2)
        A2, B2, C2 = GeneralEquation(x3, y3, x4, y4)
    m = A1 * B2 - A2 * B1
    x = (C2 * B1 - C1 * B2) / m
    y = (C1 * A2 - C2 * A1) / m
    return [x, y]


def GetDisofPointtoLine(x, y, x1, y1, x2, y2):
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    dis = (math.fabs(a * x + b * y + c)) / (math.pow(a * a + b * b, 0.5))
    return dis


def post():
    names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
             'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
             'swimming-pool', 'helicopter']

    con_thres = 0.06
    res_path = 'exp/midnet_wo_sdcn_768/result/'
    files = os.listdir(res_path)
    bar = Bar('saving...: ', max=len(files))
    outputs = [[] for i in range(15)]

    exist = [i[:-4] for i in os.listdir('midnet_wo_sdcn_wo_crop/results')]

    for file_id, file_name in enumerate(files):

        if file_name[:-4] in exist:
            Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:}' \
                .format(file_id, len(files), total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
            continue

        file_out = []
        res = np.load(res_path + file_name)[0]

        for cat_id in range(1, 16):

            center = res[0][cat_id]
            left, top, right, bottom = res[1][cat_id], res[2][cat_id], res[3][cat_id], res[4][cat_id]

            # 点的抑制与匹配
            center = [c for c in center if c[3] >= con_thres]
            left = [p for p in left if p[4] >= con_thres]
            top = [p for p in top if p[4] >= con_thres]
            right = [p for p in right if p[4] >= con_thres]
            bottom = [p for p in bottom if p[4] >= con_thres]

            objs = []
            for c in center:
                obj = [[] for i in range(5)]
                obj[0] = [c[0], c[1], c[3], c[2]]
                objs.append(obj)

            for p in left:
                x, y, w, h = p[0], p[1], p[2], p[3]
                ox, oy = x + w, y - h
                min_dis = 1024 * 1024
                for obj in objs:
                    cx, cy, r = obj[0][0], obj[0][1], obj[0][3]
                    d = dis([cx, cy], [ox, oy])
                    if d <= r and d < min_dis:
                        if len(obj[1]) == 0 or p[4] > obj[1][6]:
                            obj[1] = [x, y, w, h, ox, oy, p[4]]
                            min_dis = d

            for p in top:
                x, y, w, h = p[0], p[1], p[2], p[3]
                ox, oy = x + w, y + h
                min_dis = 1024 * 1024
                for obj in objs:
                    cx, cy, r = obj[0][0], obj[0][1], obj[0][3]
                    d = dis([cx, cy], [ox, oy])
                    if d <= r and d < min_dis:
                        if len(obj[2]) == 0 or p[4] > obj[2][6]:
                            obj[2] = [x, y, w, h, ox, oy, p[4]]
                            min_dis = d

            for p in right:
                x, y, w, h = p[0], p[1], p[2], p[3]
                ox, oy = x - w, y + h
                min_dis = 1024 * 1024
                for obj in objs:
                    cx, cy, r = obj[0][0], obj[0][1], obj[0][3]
                    d = dis([cx, cy], [ox, oy])
                    if d <= r and d < min_dis:
                        if len(obj[3]) == 0 or p[4] > obj[3][6]:
                            obj[3] = [x, y, w, h, ox, oy, p[4]]
                            min_dis = d

            for p in bottom:
                x, y, w, h = p[0], p[1], p[2], p[3]
                ox, oy = x - w, y - h
                min_dis = 1024 * 1024
                for obj in objs:
                    cx, cy, r = obj[0][0], obj[0][1], obj[0][3]
                    d = dis([cx, cy], [ox, oy])
                    if d <= r and d < min_dis:
                        if len(obj[4]) == 0 or p[4] > obj[4][6]:
                            obj[4] = [x, y, w, h, ox, oy, p[4]]
                            min_dis = d

            for obj in objs:

                if [] in obj:
                    continue

                cot = 0
                for p in obj[1:5]:
                    if p[2] <= 1 or p[3] <= 1 or 1 / 256 > abs(p[2] / p[3]) or abs(p[2] / p[3]) > 256:
                        # if abs(p[2]) < 1 or abs(p[3]) < 1:
                        cot += 1
                if cot <= 1:
                    continue

                w, h = 0, 0
                for p in obj[1:5]:
                    if abs(p[0] - obj[0][0]) > w:
                        w = abs(p[0] - obj[0][0])
                    if abs(p[1] - obj[0][1]) > h:
                        h = abs(p[1] - obj[0][1])

                corner1 = [obj[0][0] - w, obj[0][1] - h]
                corner2 = [obj[0][0] - w, obj[0][1] + h]
                corner3 = [obj[0][0] + w, obj[0][1] + h]
                corner4 = [obj[0][0] + w, obj[0][1] - h]

                con = obj[0][2]
                out = [file_name[:-4], con] + corner1 + corner2 + corner3 + corner4
                out = [str(i) for i in out]
                tmp = ''
                for i in out[:-1]:
                    tmp = tmp + i + ' '
                tmp = tmp + out[-1] + '\n'
                outputs[cat_id - 1].append(tmp)

                objs.remove(obj)

            for obj in objs:
                if [] in obj:
                    continue
                # 调整中心点
                sum_con = sum([obj[1][6], obj[2][6], obj[3][6], obj[4][6]])
                cx, cy = 0, 0
                for i in obj[1:5]:
                    cx += i[4] * i[6] / sum_con
                    cy += i[5] * i[6] / sum_con
                obj[0][0] = cx
                obj[0][1] = cy
                obj[0][2] = sum_con / 4

                # 计算斜率
                k = cal_k(obj)

                # 调整关键点
                dis1 = GetDisofPointtoLine(obj[2][0], obj[2][1], obj[0][0], obj[0][1],
                                           obj[0][0] + 128, obj[0][1] + 128 * k)
                dis2 = GetDisofPointtoLine(obj[4][0], obj[4][1], obj[0][0], obj[0][1],
                                           obj[0][0] + 128, obj[0][1] + 128 * k)
                dis0 = dis1 * obj[2][6] / (obj[2][6] + obj[4][6]) + dis2 * obj[4][6] / (obj[2][6] + obj[4][6])
                A1, B1, C1 = GeneralEquation(obj[0][0], obj[0][1], obj[0][0] + 128, obj[0][1] + 128 * -1 / k)
                A, B, C = GeneralEquation(obj[0][0], obj[0][1], obj[0][0] + 128, obj[0][1] + 128 * k)
                C2 = C - dis0 * (A ** 2 + B ** 2) ** 0.5
                C3 = C + dis0 * (A ** 2 + B ** 2) ** 0.5
                corner1 = GetIntersectPointofLines(A1=A1, B1=B1, C1=C1, A2=A, B2=B, C2=C2)
                corner2 = GetIntersectPointofLines(A1=A1, B1=B1, C1=C1, A2=A, B2=B, C2=C3)
                obj[2][0], obj[2][1] = int(corner1[0]), int(corner1[1])
                obj[4][0], obj[4][1] = int(corner2[0]), int(corner2[1])

                dis1 = GetDisofPointtoLine(obj[1][0], obj[1][1], obj[0][0], obj[0][1],
                                           obj[0][0] + 128, obj[0][1] + 128 * -1 / k)
                dis2 = GetDisofPointtoLine(obj[3][0], obj[3][1], obj[0][0], obj[0][1],
                                           obj[0][0] + 128, obj[0][1] + 128 * -1 / k)
                dis0 = dis1 * obj[1][6] / (obj[1][6] + obj[3][6]) + dis2 * obj[3][6] / (obj[1][6] + obj[3][6])
                A1, B1, C1 = GeneralEquation(obj[0][0], obj[0][1], obj[0][0] + 128, obj[0][1] + 128 * k)
                A, B, C = GeneralEquation(obj[0][0], obj[0][1], obj[0][0] + 128, obj[0][1] + 128 * -1 / k)
                C2 = C - dis0 * (A ** 2 + B ** 2) ** 0.5
                C3 = C + dis0 * (A ** 2 + B ** 2) ** 0.5
                corner1 = GetIntersectPointofLines(A1=A1, B1=B1, C1=C1, A2=A, B2=B, C2=C2)
                corner2 = GetIntersectPointofLines(A1=A1, B1=B1, C1=C1, A2=A, B2=B, C2=C3)
                obj[1][0], obj[1][1] = int(corner1[0]), int(corner1[1])
                obj[3][0], obj[3][1] = int(corner2[0]), int(corner2[1])

                # 计算角点
                corner1 = GetIntersectPointofLines(obj[1][0], obj[1][1], obj[1][0] + 128, obj[1][1] + 128 * -1 / k,
                                                   obj[2][0], obj[2][1], obj[2][0] + 128, obj[2][1] + 128 * k)
                corner2 = GetIntersectPointofLines(obj[2][0], obj[2][1], obj[2][0] + 128, obj[2][1] + 128 * k,
                                                   obj[3][0], obj[3][1], obj[3][0] + 128, obj[3][1] + 128 * -1 / k)
                corner3 = GetIntersectPointofLines(obj[3][0], obj[3][1], obj[3][0] + 128, obj[3][1] + 128 * -1 / k,
                                                   obj[4][0], obj[4][1], obj[4][0] + 128, obj[4][1] + 128 * k)
                corner4 = GetIntersectPointofLines(obj[4][0], obj[4][1], obj[4][0] + 128, obj[4][1] + 128 * k,
                                                   obj[1][0], obj[1][1], obj[1][0] + 128, obj[1][1] + 128 * -1 / k)

                con = sum([obj[0][2], obj[1][6], obj[2][6], obj[3][6], obj[4][6]]) / 5
                # con = obj[0][2]
                out = [file_name[:-4], con] + corner1 + corner2 + corner3 + corner4
                out = [str(i) for i in out]
                tmp = ''
                for i in out[:-1]:
                    tmp = tmp + i + ' '
                tmp = tmp + out[-1] + '\n'
                outputs[cat_id - 1].append(tmp)
                file_out.append(tmp[:-2] + ' %s\n' % names[cat_id - 1])

        det_txt = open('midnet_wo_sdcn_wo_crop/results/%s.txt' % file_name[:-4], 'w')
        for obj in file_out:
            det_txt.write(obj)
        det_txt.close()
        # print('midnet_wo_sdcn/results/%s.txt' % file_name[:-4])

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:}' \
            .format(file_id, len(files), total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()

    bar.finish()

    # for i, output in enumerate(outputs):
    #     det_txt = open('midnet_log/%s.txt' % names[i], 'w')
    #     for obj in output:
    #         det_txt.write(obj)
    #     det_txt.close()


if __name__ == '__main__':
    post()
