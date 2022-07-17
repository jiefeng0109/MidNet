import os

import PIL.Image as Image

path = 'results_hrsc_side_center_dcn_cat_cpools/'


def paste(png1, png2):
    img1, img2 = Image.open(png1), Image.open(png2)
    size = img1.size
    joint = Image.new('RGB', (size[0] * 2, size[1]))
    loc1, loc2 = (0, 0), (size[0], 0)
    joint.paste(img1, loc1)
    joint.paste(img2, loc2)
    return joint


img_list = os.listdir(path + 'bboxes')
for img in img_list:
    joint = paste(path + 'points/' + img, path + 'bboxes/' + img)
    joint.save(path + img)
    print(path + img)
