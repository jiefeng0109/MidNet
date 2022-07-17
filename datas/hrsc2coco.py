import glob
import json
import math
import os
import xml.etree.ElementTree as ET

load_dir = '/media/titan/E/Liang/HRSC2016/Train/Annotations'
pi = 3.1415926535

# val = []
# f = open('/media/titan/E/Liang/HRSC2016/Coco/val.txt', 'r')
# for line in f:
#     val.append(line[:-1])

# 解析文件名
xml_lists = glob.glob(load_dir + '/*.xml')
xml_basenames = []
for item in xml_lists:
    xml_basenames.append(os.path.basename(item))
xml_names = []
for item in xml_basenames:
    xml_names.append(os.path.splitext(item)[0])

coco = dict()
coco['info'] = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []
coco["licenses"] = []

info = {"year": 2021.7,
        "version": '1.0',
        "description": 'hrsc2coco',
        "contributor": 'lyp',
        "url": 'null',
        "date_created": '2021.7.1'
        }
coco['info'] = info

categories = {"id": 1,
              "name": 'ship',
              "supercategory": 'null',
              }
coco['categories'].append(categories)

item_id = 0

for id, xml in enumerate(xml_names):

    # if xml in val:
    #     continue

    tree = ET.parse(os.path.join(load_dir, str(xml) + '.xml'))
    root = tree.getroot()

    image = {"id": id,
             "width": root.find('Img_SizeWidth').text,
             "height": root.find('Img_SizeHeight').text,
             "file_name": str(xml) + '.bmp',
             "license": 0,
             "flickr_url": 'null',
             "coco_url": 'null',
             "date_captured": '2021.7.1'
             }
    coco['images'].append(image)

    for Object in root.findall('./HRSC_Objects/HRSC_Object'):
        mbox_cx = float(Object.find('mbox_cx').text)
        mbox_cy = float(Object.find('mbox_cy').text)
        mbox_w = float(Object.find('mbox_w').text)
        mbox_h = float(Object.find('mbox_h').text)
        mbox_ang = float(Object.find('mbox_ang').text)

        # 计算舰首与舰尾点坐标
        bow_x = mbox_cx + mbox_w / 2 * math.cos(mbox_ang)
        bow_y = mbox_cy + mbox_w / 2 * math.sin(mbox_ang)
        tail_x = mbox_cx - mbox_w / 2 * math.cos(mbox_ang)
        tail_y = mbox_cy - mbox_w / 2 * math.sin(mbox_ang)
        bowA_x = round(bow_x + mbox_h / 2 * math.sin(mbox_ang))
        bowA_y = round(bow_y - mbox_h / 2 * math.cos(mbox_ang))
        bowB_x = round(bow_x - mbox_h / 2 * math.sin(mbox_ang))
        bowB_y = round(bow_y + mbox_h / 2 * math.cos(mbox_ang))
        tailA_x = round(tail_x + mbox_h / 2 * math.sin(mbox_ang))
        tailA_y = round(tail_y - mbox_h / 2 * math.cos(mbox_ang))
        tailB_x = round(tail_x - mbox_h / 2 * math.sin(mbox_ang))
        tailB_y = round(tail_y + mbox_h / 2 * math.cos(mbox_ang))
        # print(bowA_x, bowA_y, bowB_x, bowB_y, tailA_x, tailA_y, tailB_x, tailB_y)

        # find extreme point
        # center = [int(i) for i in [mbox_cx, mbox_cy, mbox_w, mbox_h]]

        x = [bowA_x, tailA_x, tailB_x, bowB_x]
        y = [bowA_y, tailA_y, tailB_y, bowB_y]

        xmin, ymin, xmax, ymax = min(x), min(y), max(x), max(y)
        cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2
        w, h = xmax - xmin, ymax - ymin

        corners = []
        for i in range(4):
            corners.append([x[i], y[i]])

        sides = []
        for i in range(4):
            sides.append(
                [(corners[i][0] + corners[(i + 1) % 4][0]) / 2, (corners[i][1] + corners[(i + 1) % 4][1]) / 2])
        for i in range(4):
            sides[i] += [sides[i][0] - cx, sides[i][1] - cy]

        sides.sort(key=lambda x: (x[2], - x[3]))

        if sides[0][1] >= sides[1][1]:
            left_side, top_side = sides[0], sides[1]
        else:
            left_side, top_side = sides[1], sides[0]

        if sides[2][1] <= sides[3][1]:
            right_side, bottom_side = sides[2], sides[3]
        else:
            right_side, bottom_side = sides[3], sides[2]

        left_side = [int(i) for i in left_side]
        top_side = [int(i) for i in top_side]
        right_side = [int(i) for i in right_side]
        bottom_side = [int(i) for i in bottom_side]

        annotation = {"id": item_id,
                      "image_id": id,
                      "category_id": 1,
                      "left": left_side,
                      "top": top_side,
                      "right": right_side,
                      "bottom": bottom_side,
                      "center": [cx, cy, w, h],
                      }
        coco['annotations'].append(annotation)
        item_id += 1

    with open('/media/titan/E/Liang/HRSC2016/Coco/annotations/train.json', 'w') as f:
        json.dump(coco, f)
