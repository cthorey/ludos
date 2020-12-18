import json
import os

import numpy as np
from box import Box

from ludos.utils import viz
from PIL import Image
from tqdm import tqdm

ROOT_DIR = os.environ['ROOT_DIR']
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'interim')


class PatchDataset(object):
    def __init__(self, data_name, split):
        self.data_name = data_name
        self.split = split
        fname = os.path.join(DATA_PATH, data_name,
                             'annotations_{}.json'.format(split))
        with open(fname) as f:
            self.dataset = Box(json.load(f))
        self._index()

    def __len__(self):
        return len(self.ids)

    @property
    def weights(self):
        return self._weigths

    @property
    def sequence(self):
        return [{
            'img': self.get_img_info(idx)['image_path'],
            'seg': self.get_img_info(idx)['seg_path']
        } for idx in self.ids]

    def _index(self):
        self.ids = [d['id'] for d in self.dataset.data]
        self._weights = []
        for idx in tqdm(self.ids):
            self._weights.append(
                np.array(Image.open(self.get_img_info(idx)['seg_path'])).sum()
                > 0)

    def get_img_info(self, idx):
        return self.dataset.data[idx]

    def __getitem__(self, idx):
        info = self.dataset.data[idx]
        return Image.open(info['image_path']), Image.open(info['seg_path'])

    def display(self, idx, alpha=1):
        img, seg = self[idx]
        mask = np.expand_dims(np.array(seg), -1)
        img = viz.apply_masks(np.array(img), mask, alpha=alpha)
        return Image.fromarray(img.astype('uint8'))
