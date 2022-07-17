import os

import PIL.Image as Image
import numpy as np
from PIL import ImageDraw, ImageFont
import random
from DOTA_devkit import polyiou
import math
import pdb


def py_cpu_nms_poly_fast(dets, thresh):
    obbs = dets[:, 0:8]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


res_path = 'final/'
img_path = '/media/ubuntu/新加卷1/Liang/DOTA/test/images/'
# gt_path = '/data/DOTA/test/labelTxt/'

names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
         'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
         'swimming-pool', 'helicopter']

colors = []
for i in range(15):
    colors.append((random.randint(64, 255), random.randint(64, 255), random.randint(0, 192)))
font = ImageFont.truetype('/usr/share/fonts/truetype/Sarai/Sarai.ttf', 24)

output = [[] for i in range(15)]
for file in os.listdir(res_path):
    f = open(res_path + file)
    id = names.index(file[:-4])
    for line in f:
        output[id].append(line[:-2].split(' '))
    f.close()

files = os.listdir(img_path)

result = [[] for i in range(15)]

for file in files:

    img = Image.open(img_path + file)
    draw = ImageDraw.ImageDraw(img)

    # f = open(gt_path + file[:-4] + '.txt')
    # for line in f:
    #     if line[0] == 'i' or line[0] == 'g':
    #         continue
    #     line = line.split(' ')
    #     bbox = [int(i) for i in line[0:8]]
    #     draw.line(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=(0, 0, 255), width=3)
    #     draw.line(((bbox[2], bbox[3]), (bbox[4], bbox[5])), fill=(0, 0, 255), width=3)
    #     draw.line(((bbox[4], bbox[5]), (bbox[6], bbox[7])), fill=(0, 0, 255), width=3)
    #     draw.line(((bbox[6], bbox[7]), (bbox[0], bbox[1])), fill=(0, 0, 255), width=3)
    #     draw.text((bbox[4], bbox[5]), line[8], fill=(0, 0, 255), font=font)

    nms = []
    for cat in range(0, 15):
        preds = [pred for pred in output[cat] if pred[0] == file[:-4]]
        for pred in preds:
            bbox = [int(float(i)) for i in pred[2:]]
            nms.append(bbox + [float(pred[1]), cat])
    if len(nms) > 0:
        keep = py_cpu_nms_poly_fast(np.array(nms), 0.3)
    else:
        keep = []

    flag = 0

    for i, pred in enumerate(nms):
        # if i not in keep:
        #     continue
        bbox, score, cat = pred[0:8], pred[8], pred[9]

        # if score > 0.1 and names[cat] == 'ship':
        if score > 0.1:
            draw.line(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=colors[cat], width=3)
            draw.line(((bbox[2], bbox[3]), (bbox[4], bbox[5])), fill=colors[cat], width=3)
            draw.line(((bbox[4], bbox[5]), (bbox[6], bbox[7])), fill=colors[cat], width=3)
            draw.line(((bbox[6], bbox[7]), (bbox[0], bbox[1])), fill=colors[cat], width=3)
            draw.text((bbox[6], bbox[7]), '%s: %.2f' % (names[cat], score), fill=colors[cat], font=font)
            flag = 1
        tmp = '%s %f %d %d %d %d %d %d %d %d\n' % (
            file[:-4], score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7])
        result[cat].append(tmp)

    if flag == 1:
        img.save('midnet_wo_sdcn/images/' + file[:-4] + '.png', quality=100)
        print('midnet_wo_sdcn/images/' + file[:-4] + '.png')

# for i, output in enumerate(result):
#     det_txt = open('nms/%s.txt' % names[i], 'w')
#     for obj in output:
#         det_txt.write(obj)
#     det_txt.close()
