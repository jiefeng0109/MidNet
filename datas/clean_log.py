import os
import shutil

list = os.listdir('../exp/midnet_wo_sdcn_wo_crop/')
print(list)
for dir in list:
    if dir[0] == 'l':
        try:
            f = open('../exp/midnet_wo_sdcn_wo_crop/' + dir + '/log.txt', 'r')  # 打开label文件
            lines = f.readlines()
            if len(lines) == 0:
                shutil.rmtree('../exp/midnet_wo_sdcn_wo_crop/' + dir)
                print('../exp/midnet_wo_sdcn_wo_crop/' + dir)
        except:
            continue
