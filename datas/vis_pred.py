import os

import PIL.Image as Image
from PIL import ImageDraw

file = open('../merge/ship.txt')
preds = []
for line in file:
    line = line[:-1].split(' ')
    preds.append(line)
file.close()
img_path = '/media/titan/E/Liang/DOTA/val_ship/images/'
img_names = os.listdir(img_path)
gt_path = '/media/titan/E/Liang/DOTA/val_ship/labelTxt/'
for img_name in img_names:
    img = Image.open(img_path + img_name)
    draw = ImageDraw.ImageDraw(img)

    file = open(gt_path + img_name[:-4] + '.txt')
    objs = []
    for line in file:
        if line[0] == 'i' or line[0] == 'g':
            continue
        line = line.split(' ')
        if line[8] == 'ship':
            objs.append([float(i) for i in line[0:8]])
    file.close()
    for obj in objs:
        bbox = obj
        draw.line(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=(0, 0, 255), width=3)
        draw.line(((bbox[2], bbox[3]), (bbox[4], bbox[5])), fill=(0, 0, 255), width=3)
        draw.line(((bbox[4], bbox[5]), (bbox[6], bbox[7])), fill=(0, 0, 255), width=3)
        draw.line(((bbox[6], bbox[7]), (bbox[0], bbox[1])), fill=(0, 0, 255), width=3)

    objs = [i for i in preds if i[0] == img_name[:-4]]
    for obj in objs:
        bbox = [float(i) for i in obj[2:]]
        draw.line(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=(0, 255, 0), width=3)
        draw.line(((bbox[2], bbox[3]), (bbox[4], bbox[5])), fill=(0, 255, 0), width=3)
        draw.line(((bbox[4], bbox[5]), (bbox[6], bbox[7])), fill=(0, 255, 0), width=3)
        draw.line(((bbox[6], bbox[7]), (bbox[0], bbox[1])), fill=(0, 255, 0), width=3)

    img.save('../results_dota_side_center_cpools/' + img_name)
    print('../results_dota_side_center_cpools/' + img_name)
