import os
import shutil

img_path = '/media/titan/E/Liang/DOTA/val/images'
label_path = '/media/titan/E/Liang/DOTA/val/labelTxt'

dst_img_path = '/media/titan/E/Liang/DOTA/val_ship/images'
dst_label_path = '/media/titan/E/Liang/DOTA/val_ship/labelTxt'

label_list = os.listdir(label_path)
for label in label_list:
    save = 0
    ann = open(os.path.join(label_path, label))

    for line in ann:
        if 'ship' in line:
            save = 1
            break

    if save == 1:
        basename = label[:-4]
        src = os.path.join(img_path, basename + '.png')
        dst = os.path.join(dst_img_path, basename + '.png')
        shutil.move(src, dst)

        src = os.path.join(label_path, basename + '.txt')
        dst = os.path.join(dst_label_path, basename + '.txt')
        shutil.move(src, dst)

    print(label[:-4])
