import json
import math
import os

import PIL.Image as Image
import numpy as np
from PIL import ImageDraw


def dis(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_nms(center, left, top, right, bottom, thres=32, thres_con=0.06):
    remove = []
    center = center.tolist()
    for i in center:
        for j in center:
            if 0 < dis(i[0:2], j[0:2]) < thres:
                if i[4] > j[4]:
                    remove.append(center.index(j))
                else:
                    remove.append(center.index(i))
    temp = []
    for i in range(len(center)):
        if i not in remove and center[i][4] >= thres_con:
            x1 = center[i][0]
            y1 = center[i][1]
            x2 = center[i][2]
            y2 = center[i][3]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            temp.append([cx, cy, center[i][4]])
    center = temp

    centers = []
    for i in range(len(left)):
        x = int((left[i][0] + left[i][2]) / 2)
        y = int((left[i][1] + left[i][3]) / 2)
        w = int((left[i][2] - left[i][0]))
        h = int((left[i][3] - left[i][1]))
        cx = x + w
        cy = y - h
        centers.append([cx, cy, left[i][4]])
    remove = []
    for i in centers:
        for j in centers:
            if 0 < dis(i[0:2], j[0:2]) < thres:
                if i[2] > j[2]:
                    remove.append(centers.index(j))
                else:
                    remove.append(centers.index(i))
    temp = []
    for i in range(len(left)):
        if i not in remove and left[i][4] >= thres_con:
            x = int((left[i][0] + left[i][2]) / 2)
            y = int((left[i][1] + left[i][3]) / 2)
            w = int((left[i][2] - left[i][0]))
            h = int((left[i][3] - left[i][1]))
            cx = x + w
            cy = y - h
            temp.append([x, y, w, h, cx, cy, left[i][4]])
    left = temp

    centers = []
    for i in range(len(top)):
        x = int((top[i][0] + top[i][2]) / 2)
        y = int((top[i][1] + top[i][3]) / 2)
        w = int((top[i][2] - top[i][0]))
        h = int((top[i][3] - top[i][1]))
        cx = x + w
        cy = y + h
        centers.append([cx, cy, top[i][4]])
    remove = []
    for i in centers:
        for j in centers:
            if 0 < dis(i[0:2], j[0:2]) < thres:
                if i[2] > j[2]:
                    remove.append(centers.index(j))
                else:
                    remove.append(centers.index(i))
    temp = []
    for i in range(len(top)):
        if i not in remove and top[i][4] >= thres_con:
            x = int((top[i][0] + top[i][2]) / 2)
            y = int((top[i][1] + top[i][3]) / 2)
            w = int((top[i][2] - top[i][0]))
            h = int((top[i][3] - top[i][1]))
            cx = x + w
            cy = y + h
            temp.append([x, y, w, h, cx, cy, top[i][4]])
    top = temp

    centers = []
    for i in range(len(right)):
        x = int((right[i][0] + right[i][2]) / 2)
        y = int((right[i][1] + right[i][3]) / 2)
        w = int((right[i][2] - right[i][0]))
        h = int((right[i][3] - right[i][1]))
        cx = x - w
        cy = y + h
        centers.append([cx, cy, right[i][4]])
    remove = []
    for i in centers:
        for j in centers:
            if 0 < dis(i[0:2], j[0:2]) < thres:
                if i[2] > j[2]:
                    remove.append(centers.index(j))
                else:
                    remove.append(centers.index(i))
    temp = []
    for i in range(len(right)):
        if i not in remove and right[i][4] >= thres_con:
            x = int((right[i][0] + right[i][2]) / 2)
            y = int((right[i][1] + right[i][3]) / 2)
            w = int((right[i][2] - right[i][0]))
            h = int((right[i][3] - right[i][1]))
            cx = x - w
            cy = y + h
            temp.append([x, y, w, h, cx, cy, right[i][4]])
    right = temp

    centers = []
    for i in range(len(bottom)):
        x = int((bottom[i][0] + bottom[i][2]) / 2)
        y = int((bottom[i][1] + bottom[i][3]) / 2)
        w = int((bottom[i][2] - bottom[i][0]))
        h = int((bottom[i][3] - bottom[i][1]))
        cx = x - w
        cy = y - h
        centers.append([cx, cy, bottom[i][4]])
    remove = []
    for i in centers:
        for j in centers:
            if 0 < dis(i[0:2], j[0:2]) < thres:
                if i[2] > j[2]:
                    remove.append(centers.index(j))
                else:
                    remove.append(centers.index(i))
    temp = []
    for i in range(len(bottom)):
        if i not in remove and bottom[i][4] >= thres_con:
            x = int((bottom[i][0] + bottom[i][2]) / 2)
            y = int((bottom[i][1] + bottom[i][3]) / 2)
            w = int((bottom[i][2] - bottom[i][0]))
            h = int((bottom[i][3] - bottom[i][1]))
            cx = x - w
            cy = y - h
            temp.append([x, y, w, h, cx, cy, bottom[i][4]])
    bottom = temp

    return center, left, top, right, bottom


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


if __name__ == '__main__':
    load_path = 'exp/hrsc_side_center_dcn_cpools/result/'
    files = os.listdir(load_path)

    with open('/media/titan/E/Liang/HRSC2016/Coco/annotations/val.json', 'r', encoding='utf8') as f:
        gt = json.load(f)

    det_txt = open('ship.txt', 'w')

    for file_name in files:
        res_path = 'exp/hrsc_side_center_dcn_cpools/result/'
        img_path = '/media/titan/E/Liang/HRSC2016/FullDataSet/AllImages/'

        img_id = 0
        for img_name in gt['images']:
            if img_name['file_name'][:-4] == file_name[:-4]:
                img_id = img_name['id']

        res = np.load(res_path + file_name)[0]
        center = res[0][1]
        left, top, right, bottom = res[1][1], res[2][1], res[3][1], res[4][1]

        # 点的抑制
        center, left, top, right, bottom = point_nms(center, left, top, right, bottom)

        # print(center)
        # print(left)
        # print(top)
        # print(right)
        # print(bottom)

        # 匹配目标
        objs = []
        # for c in center:
        #     for l in left:
        #         for t in top:
        #             for r in right:
        #                 for b in bottom:
        #                     p = [c, l, t, r, b]
        #                     x_min = min(c[0], l[4], t[4], r[4], b[4])
        #                     x_max = max(c[0], l[4], t[4], r[4], b[4])
        #                     y_min = min(c[1], l[5], t[5], r[5], b[5])
        #                     y_max = max(c[1], l[5], t[5], r[5], b[5])
        #                     if 0 < x_max - x_min < 32 and 0 < y_max - y_min < 32:
        #                         objs.append([c, l, t, r, b])
        for l in left:
            for t in top:
                for r in right:
                    for b in bottom:
                        p = [[], l, t, r, b]
                        x_min = min(l[4], t[4], r[4], b[4])
                        x_max = max(l[4], t[4], r[4], b[4])
                        y_min = min(l[5], t[5], r[5], b[5])
                        y_max = max(l[5], t[5], r[5], b[5])
                        if 0 < x_max - x_min < 32 and 0 < y_max - y_min < 32:
                            objs.append([[0, 0, 0], l, t, r, b])

        img = Image.open(img_path + file_name[:-4] + '.bmp')
        draw = ImageDraw.ImageDraw(img)
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
                  [255, 114, 86], [180, 82, 205], [255, 20, 147], [255, 165, 0], [255, 105, 180], [0, 191, 255]]

        gt_txt = open('gt/' + file_name[:-4] + '.txt', 'w')
        for ann in [ann for ann in gt['annotations'] if ann["image_id"] == img_id]:
            color = [0, 0, 255]

            draw.line((tuple(ann['left_corner']), tuple(ann['top_corner'])), fill=tuple(color), width=3)
            draw.line((tuple(ann['top_corner']), tuple(ann['right_corner'])), fill=tuple(color), width=3)
            draw.line((tuple(ann['right_corner']), tuple(ann['bottom_corner'])), fill=tuple(color), width=3)
            draw.line((tuple(ann['bottom_corner']), tuple(ann['left_corner'])), fill=tuple(color), width=3)

            out = ann['left_corner'] + ann['top_corner'] + ann['right_corner'] + ann['bottom_corner'] + ['ship', 0]
            out = [str(i) for i in out]
            for i in out[:-1]:
                gt_txt.write(i + ' ')
            gt_txt.write(out[-1] + '\n')

        gt_txt.close()
        for obj in objs:
            # color = random.choice(colors)
            # colors.remove(color)
            color = [0, 255, 0]

            # 调整中心点
            # sum_con = sum([obj[0][2], obj[1][6], obj[2][6], obj[3][6], obj[4][6]])
            # cx, cy = obj[0][0] * obj[0][2] / sum_con, obj[0][1] * obj[0][2] / sum_con
            sum_con = sum([obj[1][6], obj[2][6], obj[3][6], obj[4][6]])
            cx, cy = 0, 0
            for i in obj[1:5]:
                cx += i[4] * i[6] / sum_con
                cy += i[5] * i[6] / sum_con
            obj[0][0] = cx
            obj[0][1] = cy
            obj[0][2] = sum_con / 4

            # 点的可视化
            draw.ellipse(((cx - 6, cy - 6), (cx + 6, cy + 6)), fill=tuple(color), outline=tuple(color),
                         width=1)
            for point in obj[1:5]:
                cx = point[0]
                cy = point[1]
                draw.ellipse(((cx - 6, cy - 6), (cx + 6, cy + 6)), fill=tuple(color), outline=tuple(color),
                             width=1)

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
            con = sum([obj[0][2], obj[1][6], obj[2][6], obj[3][6], obj[4][6]]) / 5
            out = [file_name[:-4], con] + corner1 + corner2 + corner3 + corner4
            out = [str(i) for i in out]
            for i in out[:-1]:
                det_txt.write(i + ' ')
            det_txt.write(out[-1] + '\n')

        img.save('results_side_cpools/bboxes/' + file_name[:-4] + '.png')
        print('results_side_cpools/bboxes/' + file_name[:-4] + '.png')

        # break

    det_txt.close()
