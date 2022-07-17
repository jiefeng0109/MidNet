import json
import os

import cv2

import dota_utils as util


def DOTA2COCO(srcpath, destfile):
    wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
                   'ship', 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
                   'harbor', 'swimming-pool', 'helicopter']

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    info = {'contributor': 'captain group',
            'data_created': '2018',
            'description': 'This is 1.0 version of DOTA dataset.',
            'url': 'http://captain.whu.edu.cn/DOTAweb/',
            'version': '1.0',
            'year': 2018}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(wordname_15):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)

            # if basename != 'P0047_102_1143':
            #     continue

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                single_obj = {}
                single_obj['category_id'] = wordname_15.index(obj['name']) + 1
                single_obj['iscrowd'] = 0

                x = obj['poly'][0::2]
                y = obj['poly'][1::2]

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

                single_obj['center'] = [cx, cy, w, h]
                single_obj['left'] = left_side
                single_obj['top'] = top_side
                single_obj['right'] = right_side
                single_obj['bottom'] = bottom_side

                single_obj['image_id'] = image_id
                single_obj['id'] = inst_count
                single_obj['poly'] = obj['poly']

                data_dict['annotations'].append(single_obj)

                inst_count = inst_count + 1
            image_id = image_id + 1
            if image_id % 500 == 0:
                print(image_id, imagepath)
        json.dump(data_dict, f_out)


if __name__ == '__main__':
    DOTA2COCO('/media/ubuntu/新加卷/Liang/DOTA/trainval_768/',
              '/media/ubuntu/新加卷/Liang/DOTA/COCO/annotations/train.json')
