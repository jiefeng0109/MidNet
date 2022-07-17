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
        filenames = os.listdir(srcpath)
        print(filenames)
        for file in filenames:
            imagepath = os.path.join(srcpath, file)
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = file
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
            # if image_id % 500 == 0:
            #     print(image_id, imagepath)
        json.dump(data_dict, f_out)


if __name__ == '__main__':
    DOTA2COCO('/media/ubuntu/新加卷/Liang/DOTA/test_768/',
              '/media/ubuntu/新加卷/Liang/DOTA/COCO/annotations/val.json')
