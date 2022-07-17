# -*- coding: cp936 -*-
import math
import numpy as np


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


def GetAreaOfPolyGon(points):
    area = 0
    if (len(points) < 3):
        raise Exception("error")

    p1 = points[0]
    for i in range(1, len(points) - 1):
        p2 = points[1]
        p3 = points[2]

        # 计算向量
        vecp1p2 = Point(p2.x - p1.x, p2.y - p1.y)
        vecp2p3 = Point(p3.x - p2.x, p3.y - p2.y)

        # 判断顺时针还是逆时针，顺时针面积为正，逆时针面积为负
        vecMult = vecp1p2.x * vecp2p3.y - vecp1p2.y * vecp2p3.x  # 判断正负方向比较有意思
        sign = 0
        if (vecMult > 0):
            sign = 1
        elif (vecMult < 0):
            sign = -1

        triArea = GetAreaOfTriangle(p1, p2, p3) * sign
        area += triArea
    return abs(area)


def GetAreaOfTriangle(p1, p2, p3):
    '''计算三角形面积   海伦公式'''
    area = 0
    p1p2 = GetLineLength(p1, p2)
    p2p3 = GetLineLength(p2, p3)
    p3p1 = GetLineLength(p3, p1)
    s = (p1p2 + p2p3 + p3p1) / 2
    area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)  # 海伦公式
    area = math.sqrt(area)
    return area


def GetLineLength(p1, p2):
    '''计算边长'''
    length = math.pow((p1.x - p2.x), 2) + math.pow((p1.y - p2.y), 2)  # pow  次方
    length = math.sqrt(length)
    return length


def main():
    names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
             'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
             'swimming-pool', 'helicopter']

    for cat in names:
        dets = np.loadtxt('merge/%s.txt' % cat, delimiter=' ', dtype=str).tolist()
        saved = []
        for det in dets:
            try:
                p1 = Point(float(det[2]), float(det[3]))
                p2 = Point(float(det[4]), float(det[5]))
                p3 = Point(float(det[6]), float(det[7]))
                p4 = Point(float(det[8]), float(det[9]))
                points = [p1, p2, p3, p4]
                area = GetAreaOfPolyGon(points)
                if math.ceil(area) >= 10:
                    saved.append(det)
                else:
                    print(det)
            except:
                print(det)
        det_txt = open('final/%s.txt' % cat, 'w')
        for obj in saved:
            out = ''
            for item in obj[:-1]:
                out = out + item + ' '
            out = out + obj[-1] + '\n'
            det_txt.write(out)
        det_txt.close()


if __name__ == '__main__':
    main()
