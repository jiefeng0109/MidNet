import os
import xml.etree.ElementTree as ET
import math
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import glob

load_dir = '/media/titan/E/Liang/HRSC2016/Train/Annotations'
save = 'text/'
pi = 3.1415926535

# 解析文件名
xml_lists = glob.glob(load_dir + '/*.xml')

xml_basenames = []
for item in xml_lists:
    xml_basenames.append(os.path.basename(item))

xml_names = []
for item in xml_basenames:
    xml_names.append(os.path.splitext(item)[0])

for it in xml_names:
    tree = ET.parse(os.path.join(load_dir, str(it) + '.xml'))
    root = tree.getroot()

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'train_images'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(it) + '.bmp'
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = root.find('Img_SizeWidth').text
    node_height = SubElement(node_size, 'height')
    node_height.text = root.find('Img_SizeHeight').text
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = root.find('Img_SizeDepth').text

    for Object in root.findall('./HRSC_Objects/HRSC_Object'):
        mbox_cx = float(Object.find('mbox_cx').text)
        mbox_cy = float(Object.find('mbox_cy').text)
        mbox_w = float(Object.find('mbox_w').text)
        mbox_h = float(Object.find('mbox_h').text)
        mbox_ang = float(Object.find('mbox_ang').text)

        # 计算舰首 与舰尾点坐标
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
        print(bowA_x, bowA_y, bowB_x, bowB_y, tailA_x, tailA_y, tailB_x, tailB_y)

        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'text'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')

        node_x1 = SubElement(node_bndbox, 'x1')
        node_x1.text = str(bowA_x)
        node_y1 = SubElement(node_bndbox, 'y1')
        node_y1.text = str(bowA_y)

        node_x2 = SubElement(node_bndbox, 'x2')
        node_x2.text = str(bowB_x)
        node_y2 = SubElement(node_bndbox, 'y2')
        node_y2.text = str(bowB_y)

        node_x3 = SubElement(node_bndbox, 'x3')
        node_x3.text = str(tailA_x)
        node_y3 = SubElement(node_bndbox, 'y3')
        node_y3.text = str(tailA_y)

        node_x4 = SubElement(node_bndbox, 'x4')
        node_x4.text = str(tailB_x)
        node_y4 = SubElement(node_bndbox, 'y4')
        node_y4.text = str(tailB_y)

        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(min(bowA_x, bowB_x, tailA_x, tailB_x))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(min(bowA_y, bowB_y, tailA_y, tailB_y))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(max(bowA_x, bowB_x, tailA_x, tailB_x))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(max(bowA_y, bowB_y, tailA_y, tailB_y))

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    dom = parseString(xml)
    fw = open(os.path.join(save, str(it) + '.xml'), 'wb')
    fw.write(xml)
    fw.close()




