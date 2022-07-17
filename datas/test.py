import json
import os

import PIL.Image as Image
from PIL import ImageDraw

import random

with open('/media/admin/E/Liang/FGSD2021/COCO/annotations/train.json', 'r', encoding='utf8') as f:
    json_data = json.load(f)

line = True

for i in range(0, 100):
    img_id = random.randint(1, 100)
    img_name = [img['file_name'] for img in json_data['images'] if img['id'] == img_id]
    if len(img_name) == 0:
        continue
    else:
        img_name = img_name[0]

    img = Image.open('/media/admin/E/Liang/FGSD2021/COCO/train/' + img_name)
    draw = ImageDraw.ImageDraw(img)

    items = [item for item in json_data['annotations'] if item['image_id'] == img_id]

    for item in items:
        # print(item)

        # draw.line(((point[0], point[1]), (point[2], point[3])), fill=(255, 0, 255), width=3)
        # draw.line(((point[2], point[3]), (point[4], point[5])), fill=(255, 0, 255), width=3)
        # draw.line(((point[4], point[5]), (point[6], point[7])), fill=(255, 0, 255), width=3)
        # draw.line(((point[6], point[7]), (point[0], point[1])), fill=(255, 0, 255), width=3)

        coord = item['bottom']
        x = coord[0]
        y = coord[1]
        draw.ellipse(((x - 3, y - 3), (x + 3, y + 3)), fill=(255, 255, 0), outline=(255, 255, 0), width=1)
        if line:
            w = coord[2]
            h = coord[3]
            draw.line(((x, y), (x - w, y - h)), fill=(255, 255, 0), width=3)

        coord = item['right']
        x = coord[0]
        y = coord[1]
        draw.ellipse(((x - 3, y - 3), (x + 3, y + 3)), fill=(0, 0, 255), outline=(0, 0, 255), width=1)
        if line:
            w = coord[2]
            h = coord[3]
            draw.line(((x, y), (x - w, y - h)), fill=(0, 0, 255), width=3)

        coord = item['top']
        x = coord[0]
        y = coord[1]
        draw.ellipse(((x - 3, y - 3), (x + 3, y + 3)), fill=(0, 255, 0), outline=(0, 255, 0), width=1)
        if line:
            w = coord[2]
            h = coord[3]
            draw.line(((x, y), (x - w, y - h)), fill=(0, 255, 0), width=3)

        coord = item['left']
        x = coord[0]
        y = coord[1]
        draw.ellipse(((x - 3, y - 3), (x + 3, y + 3)), fill=(255, 0, 0), outline=(255, 0, 0), width=1)
        if line:
            w = coord[2]
            h = coord[3]
            draw.line(((x, y), (x - w, y - h)), fill=(255, 0, 0), width=3)

        coord = item['center']
        x = coord[0]
        y = coord[1]
        draw.ellipse(((x - 3, y - 3), (x + 3, y + 3)), fill=(0, 255, 255), outline=(0, 255, 255), width=1)

    img.save('test/' + img_name)
    print('test/' + img_name)

# img_dir = '/media/admin/E/Liang/FGSD2021/DOTA_train_resize/images/'
# label_dir = '/media/admin/E/Liang/FGSD2021/DOTA_train_resize/labelTxt/'
#
# for i in range(0, 100):
#
#     img_name = random.choice(os.listdir(img_dir))
#     img = Image.open(img_dir + img_name)
#     draw = ImageDraw.ImageDraw(img)
#
#     label = open(label_dir + img_name[:-4] + '.txt')
#
#     for line in label:
#         point = [int(float(i)) for i in line.split(' ')[0:8]]
#         draw.line(((point[0], point[1]), (point[2], point[3])), fill=(255, 0, 255), width=3)
#         draw.line(((point[2], point[3]), (point[4], point[5])), fill=(255, 0, 255), width=3)
#         draw.line(((point[4], point[5]), (point[6], point[7])), fill=(255, 0, 255), width=3)
#         draw.line(((point[6], point[7]), (point[0], point[1])), fill=(255, 0, 255), width=3)
#
#     img.save('test/' + img_name)
#     print('test/' + img_name)
