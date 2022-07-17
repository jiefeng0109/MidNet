import os

names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
         'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
         'swimming-pool', 'helicopter']

outputs = [[] for i in range(15)]

file_list = os.listdir('midnet_wo_sdcn_wo_crop/results')

for i, file in enumerate(file_list):
    f = open('midnet_wo_sdcn_wo_crop/results/%s' % file, 'r')
    for line in f:
        cat_id = names.index(line.split(' ')[-1][:-1])
        outputs[cat_id].append(line[:int(-1 * line[::-1].find(' ')) - 1] + '\n')
    if i % 1000 == 0:
        print(i)

for i, output in enumerate(outputs):
    det_txt = open('ori/%s.txt' % names[i], 'w')
    for obj in output:
        det_txt.write(obj)
    det_txt.close()
