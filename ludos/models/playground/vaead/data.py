import json
import os

import numpy as np
from box import Box

from ludos.utils import viz
from PIL import Image
from tqdm import tqdm

ROOT_DIR = os.environ['ROOT_DIR']
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'interim')


class Dataset(object):
    def __init__(self, data_name, split):
        self.data_name = data_name
        self.split = split
        self.ann_file = os.path.join(DATA_PATH, data_name,
                                     'annotations_{}.json'.format(split))
        with open(self.ann_file) as f:
            self.dataset = Box(json.load(f))
        self._index()

    def __len__(self):
        return len(self.ids)

    def _index(self):
        self.ids = [d['id'] for d in self.dataset.data]

    def get_img_info(self, idx):
        return self.dataset.data[idx]

    def __getitem__(self, idx):
        info = self.dataset.data[idx]
        return Image.open(info['image_path'])

    def display(self, idx, alpha=1):
        img = self[idx]
        return Image.fromarray(img.astype('uint8'))
