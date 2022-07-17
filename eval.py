import os

from opts import opts
from test import test
from post_center_dota_wo_img import post

opt = opts().parse()

opt.load_model = 'exp/midnet_log/model_last.pth'
for file in os.listdir('exp/midnet_log/result'):
    os.remove(os.path.join('exp/midnet_log/result', file))
test(opt)
post()
