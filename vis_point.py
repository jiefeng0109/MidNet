import os

import PIL.Image as Image
import numpy as np
from PIL import ImageDraw

load_path = '/media/titan/E/Liang/CenpantalNet/exp/hrsc_side_center_dcn_cat_cpools/result'
files = os.listdir(load_path)

for file in files:
    res = np.load('/media/titan/E/Liang/CenpantalNet/exp/hrsc_side_center_dcn_cat_cpools/result/' + file)[0]
    center = res[0][1]
    left, top, right, bottom = res[1][1], res[2][1], res[3][1], res[4][1]

    img = Image.open('/media/titan/E/Liang/HRSC2016/FullDataSet/AllImages/' + file[:-4] + '.bmp')
    draw = ImageDraw.ImageDraw(img)

    thres = 0.01

    for c in center:
        color = [0, 0, 255]
        x1 = c[0]
        y1 = c[1]
        x2 = c[2]
        y2 = c[3]
        r = (x2 - x1) / 2
        cx, cy = (x2 + x1) / 2, (y2 + y1) / 2
        if c[4] > thres:
            draw.ellipse(((cx - r * 1.2, cy - r * 1.2), (cx + r * 1.2, cy + r * 1.2)), outline=(0, 255, 255), width=3)

    for i in range(len(left)):
        x1 = left[i][0]
        y1 = left[i][1]
        x2 = left[i][2]
        y2 = left[i][3]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        w = int((x2 - x1))
        h = int((y2 - y1))
        if left[i][4] > thres:
            draw.ellipse(((cx - 6, cy - 6), (cx + 6, cy + 6)), fill=(255, 0, 0), outline=(255, 0, 0), width=1)
            draw.line(((cx, cy), (cx + w, cy - h)), fill=(255, 0, 0), width=3)

    for i in range(len(left)):
        x1 = left[i][0]
        y1 = left[i][1]
        x2 = left[i][2]
        y2 = left[i][3]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        w = int((x2 - x1))
        h = int((y2 - y1))
        if left[i][4] > thres:
            draw.ellipse(((cx - 6, cy - 6), (cx + 6, cy + 6)), fill=(255, 0, 0), outline=(255, 0, 0), width=1)
            draw.line(((cx, cy), (cx + w, cy - h)), fill=(255, 0, 0), width=3)

    for i in range(len(top)):
        x1 = top[i][0]
        y1 = top[i][1]
        x2 = top[i][2]
        y2 = top[i][3]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        w = int((x2 - x1))
        h = int((y2 - y1))
        if top[i][4] > thres:
            draw.ellipse(((cx - 6, cy - 6), (cx + 6, cy + 6)), fill=(0, 255, 0), outline=(0, 255, 0), width=1)
            draw.line(((cx, cy), (cx + w, cy + h)), fill=(0, 255, 0), width=3)

    for i in range(len(right)):
        x1 = right[i][0]
        y1 = right[i][1]
        x2 = right[i][2]
        y2 = right[i][3]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        w = int((x2 - x1))
        h = int((y2 - y1))
        if right[i][4] > thres:
            draw.ellipse(((cx - 6, cy - 6), (cx + 6, cy + 6)), fill=(0, 0, 255), outline=(0, 0, 255), width=1)
            draw.line(((cx, cy), (cx - w, cy + h)), fill=(0, 0, 255), width=3)

    for i in range(len(bottom)):
        x1 = bottom[i][0]
        y1 = bottom[i][1]
        x2 = bottom[i][2]
        y2 = bottom[i][3]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        w = int((x2 - x1))
        h = int((y2 - y1))
        if bottom[i][4] > thres:
            draw.ellipse(((cx - 6, cy - 6), (cx + 6, cy + 6)), fill=(255, 255, 0), outline=(255, 255, 0), width=1)
            draw.line(((cx, cy), (cx - w, cy - h)), fill=(255, 255, 0), width=3)

    img.save('results_hrsc_side_center_dcn_cat_cpools/points/' + file[:-4] + '.png')
    print('results_hrsc_side_center_dcn_cat_cpools/points/' + file[:-4] + '.png')
