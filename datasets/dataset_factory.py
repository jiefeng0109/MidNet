from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CTDetDataset
from .coco import COCO
from .hrsc import HRSC
from .dota import DOTA
from .fgsd import FGSD
from .dota_ship import DOTA_SHIP

dataset_factory = {
    'coco': COCO,
    'hrsc': HRSC,
    'dota': DOTA,
    'fgsd': FGSD
}

_sample_factory = {
    'ctdet': CTDetDataset,
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
