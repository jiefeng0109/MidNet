import os

import dota_utils as util


def DOTA2COCO(srcpath, destfile):
    labelparent = os.path.join(srcpath, 'labelTxt')
    filenames = util.GetFileFromThisRootDir(labelparent)
    max_obj = 0
    max_name = ''
    for file in filenames:
        # annotations
        objects = util.parse_dota_poly2(file)
        if len(objects) > max_obj:
            max_obj = len(objects)
            max_name = file
    return max_obj, max_name


if __name__ == '__main__':
    n, name = DOTA2COCO('/media/titan/E/Liang/DOTA/train_crop/',
                        '/media/titan/E/Liang/DOTA/Coco/annotations/train.json')
    print(n, name)
