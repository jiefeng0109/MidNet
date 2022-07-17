import os
import shutil


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def cleandata(path, img_path, blank_label_path, blank_img_path, ext):
    name = custombasename(path)  # 名称
    f_in = open(path, 'r')  # 打开label文件
    lines = f_in.readlines()
    if len(lines) == 0:  # 如果为空
        f_in.close()
        image_path = os.path.join(img_path, name + ext)  # 样本图片的名称
        shutil.move(image_path, blank_img_path)  # 移动该样本图片到blank_img_path
        shutil.move(path, blank_label_path)  # 移动该样本图片的标签到blank_label_path
    print(path)


if __name__ == '__main__':
    root = 'E:/Liang/DOTA/val_768'
    img_path = os.path.join(root, 'images')  # 分割后的样本集
    label_path = os.path.join(root, 'labelTxt')  # 分割后的标签
    ext = '.png'  # 图片的后缀
    # 空白的样本及标签
    blank_img_path = os.path.join(root, 'blank_images')
    blank_label_path = os.path.join(root, 'blank_labelTxt')
    if not os.path.exists(blank_img_path):
        os.makedirs(blank_img_path)
    if not os.path.exists(blank_label_path):
        os.makedirs(blank_label_path)

    label_list = GetFileFromThisRootDir(label_path)
    for path in label_list:
        cleandata(path, img_path, blank_label_path, blank_img_path, ext)
