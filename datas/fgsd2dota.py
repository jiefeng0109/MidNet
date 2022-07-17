# 导入相关的包
import os
import xml.etree.ElementTree as ET
import math

load_dir = '/media/admin/E/Liang/FGSD2021/train/labels/'
save_dir = '/media/admin/E/Liang/FGSD2021/DOTA_train/labelTxt/'

names = ['航母', '黄蜂级', '塔瓦拉级', '奥斯汀级', '惠特贝岛级', '圣安东尼奥级', '新港级', '提康德罗加级', '阿利·伯克级', '佩里级',
         '刘易斯和克拉克级', '供应级', '凯泽级', '霍普级', '仁慈级', '自由级', '独立级', '复仇者级', '潜艇', '其他']
names_en = ['Aircraft-carriers', 'Wasp-class', 'Tarawa-class', 'Austin-class', 'Whidbey-Island-class',
            'San-Antonio-class', 'Newport-class', 'Ticonderoga-class', 'Arleigh-Burke-class', 'Perry-class',
            'Lewis-and-Clark-class', 'Supply-class', 'Henry-J.-Kaiser-class', 'Bob-Hope-Class', 'Mercy-class',
            'Freedom-class', 'Independence-class', 'Avenger-class', 'Submarine', 'Other']

xml_names = os.listdir(load_dir)
for xml in xml_names:
    tree = ET.parse(os.path.join(load_dir, xml))
    root = tree.getroot()
    f = open(save_dir + '%s.txt' % xml[:-4], 'w')
    f.write('imagesource:GoogleEarth\n')
    f.write('gsd:1\n')
    objs = root.findall('object')
    for obj in objs:
        cat = names_en[names.index(obj.find('name').text)]

        mbox_cx = float(obj.find('robndbox/cx').text)
        mbox_cy = float(obj.find('robndbox/cy').text)
        mbox_w = float(obj.find('robndbox/w').text)
        mbox_h = float(obj.find('robndbox/h').text)
        mbox_ang = float(obj.find('robndbox/angle').text)

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

        out = '%d %d %d %d %d %d %d %d %s 0\n' % (
            bowA_x, bowA_y, tailA_x, tailA_y, tailB_x, tailB_y, bowB_x, bowB_y, cat)
        f.write(out)

    f.close()
    print(xml)
