import shutil

f = open('/media/titan/E/Liang/HRSC2016/Coco/val.txt', 'r')
for line in f:
    ori = '/media/titan/E/Liang/HRSC2016/Coco/train/' + line[:-1] + '.bmp'
    dst = '/media/titan/E/Liang/HRSC2016/Coco/val_0/' + line[:-1] + '.bmp'
    shutil.move(ori, dst)
