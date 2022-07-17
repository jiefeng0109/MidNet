import math
import os

import PIL.Image as Image
import numpy as np
from PIL import ImageDraw


def dis(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cal_k(obj):
    ks = np.arange(-100, 100, 0.01)
    x0, y0 = obj[0][0], obj[0][1]
    x1, y1, c1 = obj[1][0], obj[1][1], obj[1][6]
    x2, y2, c2 = obj[2][0], obj[2][1], obj[2][6]
    x3, y3, c3 = obj[3][0], obj[3][1], obj[3][6]
    x4, y4, c4 = obj[4][0], obj[4][1], obj[4][6]

    sum_con = sum([obj[1][6], obj[2][6], obj[3][6], obj[4][6]])

    dis = []
    for k in ks:
        d1 = (k * x1 - y1 - k * x0 + y0) ** 2 / (k ** 2 + 1)
        d2 = (k * x3 - y3 - k * x0 + y0) ** 2 / (k ** 2 + 1)
        d3 = (-1 / k * x2 - y2 + 1 / k * x0 + y0) ** 2 / (1 / k ** 2 + 1)
        d4 = (-1 / k * x4 - y4 + 1 / k * x0 + y0) ** 2 / (1 / k ** 2 + 1)
        d = d1 * c1 / sum_con + d2 * c2 / sum_con + d3 * c3 / sum_con + d4 * c4 / sum_con
        dis.append(d)

    return ks[dis.index(min(dis))]


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


def post(epoch):
    con_thres = 0.01
    files = os.listdir('exp/hrsc_side_center_dcn_cat_cpools/result/')
    det_txt = open('results_hrsc_side_center_dcn_cat_cpools/ship.txt', 'w')

    set = ['100000672', '100000889', '100000890', '100001440', '100001505', '100001533']
    # set = ['100000794', '100000854', '100001065', '100001343']

    for file_name in files:
        # if file_name[:-4] not in set:
        #     continue
        res_path = 'exp/hrsc_side_center_dcn_cat_cpools/result/'
        img_path = '/media/titan/E/Liang/HRSC2016/Coco/val/'
        img = Image.open(img_path + file_name[:-4] + '.bmp')
        draw = ImageDraw.ImageDraw(img)

        gt_txt = open('gt/' + file_name[:-4] + '.txt', 'r')
        for line in gt_txt:
            bbox = [int(i) for i in line.split(' ')[:8]]
            color = [0, 0, 255]
            draw.line(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=tuple(color), width=3)
            draw.line(((bbox[2], bbox[3]), (bbox[4], bbox[5])), fill=tuple(color), width=3)
            draw.line(((bbox[4], bbox[5]), (bbox[6], bbox[7])), fill=tuple(color), width=3)
            draw.line(((bbox[6], bbox[7]), (bbox[0], bbox[1])), fill=tuple(color), width=3)
        gt_txt.close()

        res = np.load(res_path + file_name)[0]
        cat_id = 1
        center = res[0][cat_id]
        left, top, right, bottom = res[1][cat_id], res[2][cat_id], res[3][cat_id], res[4][cat_id]

        # 点的抑制与匹配
        center = [c for c in center if c[4] >= con_thres]
        left = [p for p in left if p[4] >= con_thres]
        top = [p for p in top if p[4] >= con_thres]
        right = [p for p in right if p[4] >= con_thres]
        bottom = [p for p in bottom if p[4] >= con_thres]
        objs = []
        for c in center:
            obj = [[] for i in range(5)]
            r = (c[3] - c[1]) * 0.8
            cx, cy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
            obj[0] = [cx, cy, c[4], r]
            objs.append(obj)

        for p in left:
            x, y = int((p[0] + p[2]) / 2), int((p[1] + p[3]) / 2)
            w, h = int((p[2] - p[0])), int((p[3] - p[1]))
            ox, oy = x + w, y - h
            min_dis = 1024 * 1024
            for obj in objs:
                cx, cy, r = obj[0][0], obj[0][1], obj[0][3]
                if dis([cx, cy], [ox, oy]) <= r and dis([cx, cy], [ox, oy]) < min_dis:
                    if len(obj[1]) == 0 or p[4] > obj[1][6]:
                        obj[1] = [x, y, w, h, ox, oy, p[4]]
                        min_dis = dis([cx, cy], [ox, oy])

        for p in top:
            x, y = int((p[0] + p[2]) / 2), int((p[1] + p[3]) / 2)
            w, h = int((p[2] - p[0])), int((p[3] - p[1]))
            ox, oy = x + w, y + h
            min_dis = 1024 * 1024
            for obj in objs:
                cx, cy, r = obj[0][0], obj[0][1], obj[0][3]
                if dis([cx, cy], [ox, oy]) <= r and dis([cx, cy], [ox, oy]) < min_dis:
                    if len(obj[2]) == 0 or p[4] > obj[2][6]:
                        obj[2] = [x, y, w, h, ox, oy, p[4]]
                        min_dis = dis([cx, cy], [ox, oy])

        for p in right:
            x, y = int((p[0] + p[2]) / 2), int((p[1] + p[3]) / 2)
            w, h = int((p[2] - p[0])), int((p[3] - p[1]))
            ox, oy = x - w, y + h
            min_dis = 1024 * 1024
            for obj in objs:
                cx, cy, r = obj[0][0], obj[0][1], obj[0][3]
                if dis([cx, cy], [ox, oy]) <= r and dis([cx, cy], [ox, oy]) < min_dis:
                    if len(obj[3]) == 0 or p[4] > obj[3][6]:
                        obj[3] = [x, y, w, h, ox, oy, p[4]]
                        min_dis = dis([cx, cy], [ox, oy])

        for p in bottom:
            x, y = int((p[0] + p[2]) / 2), int((p[1] + p[3]) / 2)
            w, h = int((p[2] - p[0])), int((p[3] - p[1]))
            ox, oy = x - w, y - h
            min_dis = 1024 * 1024
            for obj in objs:
                cx, cy, r = obj[0][0], obj[0][1], obj[0][3]
                if dis([cx, cy], [ox, oy]) <= r and dis([cx, cy], [ox, oy]) < min_dis:
                    if len(obj[4]) == 0 or p[4] > obj[4][6]:
                        obj[4] = [x, y, w, h, ox, oy, p[4]]
                        min_dis = dis([cx, cy], [ox, oy])

        color = [0, 255, 0]
        for obj in objs:

            if [] in obj:
                continue

            cot = 0
            for p in obj[1:5]:
                # if p[2] == 0 or p[3] == 0 or 1 / 512 > abs(p[2]) or abs(p[3]) > 512:
                if abs(p[2]) < 2 or abs(p[3]) < 2:
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

            draw.line((tuple(corner1), tuple(corner2)), fill=tuple(color), width=3)
            draw.line((tuple(corner2), tuple(corner3)), fill=tuple(color), width=3)
            draw.line((tuple(corner3), tuple(corner4)), fill=tuple(color), width=3)
            draw.line((tuple(corner4), tuple(corner1)), fill=tuple(color), width=3)

            con = obj[0][2]
            out = [file_name[:-4], con] + corner1 + corner2 + corner3 + corner4
            out = [str(i) for i in out]
            for i in out[:-1]:
                det_txt.write(i + ' ')
            det_txt.write(out[-1] + '\n')

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

            # 点的可视化
            # draw.ellipse(((cx - 3, cy - 3), (cx + 3, cy + 3)), fill=tuple(color), outline=tuple(color),
            #              width=3)
            # for point in obj[1:5]:
            #     cx = point[0]
            #     cy = point[1]
            #     draw.ellipse(((cx - 3, cy - 3), (cx + 3, cy + 3)), fill=tuple(color), outline=tuple(color),
            #                  width=3)

            # 计算斜率
            k = cal_k(obj)

            # 斜率的可视化
            # cx = obj[0][0]
            # cy = obj[0][1]
            # draw.line(((cx - 200, cy - 200 * k), (cx + 200, cy + 200 * k)), fill=tuple(color), width=3)
            # draw.line(((cx - 200, cy - 200 * -1 / k), (cx + 200, cy + 200 * -1 / k)), fill=tuple(color), width=3)

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

            draw.line((tuple(corner1), tuple(corner2)), fill=tuple(color), width=3)
            draw.line((tuple(corner2), tuple(corner3)), fill=tuple(color), width=3)
            draw.line((tuple(corner3), tuple(corner4)), fill=tuple(color), width=3)
            draw.line((tuple(corner4), tuple(corner1)), fill=tuple(color), width=3)

            # con = sum([obj[0][2], obj[1][6], obj[2][6], obj[3][6], obj[4][6]]) / 5
            con = obj[0][2]
            out = [file_name[:-4], con] + corner1 + corner2 + corner3 + corner4
            out = [str(i) for i in out]
            for i in out[:-1]:
                det_txt.write(i + ' ')
            det_txt.write(out[-1] + '\n')

        img.save('results_hrsc_side_center_dcn_cat_cpools/bboxes/' + file_name[:-4] + '.png')
        print('results_hrsc_side_center_dcn_cat_cpools/bboxes/' + file_name[:-4] + '.png')

        # break

    det_txt.close()


if __name__ == '__main__':
    post(250)
