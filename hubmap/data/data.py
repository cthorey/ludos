import json
import os

import numpy as np
from box import Box

from PIL import Image

ROOT_DIR = os.environ['ROOT_DIR']
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'interim')


def apply_masks(image, mask, alpha=0.5):
    """
    Blend the mask within the images - #vectorize
    """
    m = np.stack([mask] * 4).transpose(1, 2, 0) * np.array([0, 255, 0, 1])
    mask_img = (m * alpha + image * (1 - alpha)).astype('uint8')
    img = mask_img * (m != 0) + image * (m == 0)
    return img


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

    def _index(self):
        self.ids = [d['id'] for d in self.dataset.data]

    def get_img_info(self, idx):
        return self.dataset.data[idx]

    def __getitem__(self, idx):
        info = self.dataset.data[idx]
        return Image.open(info['image_path']), Image.open(info['seg_path'])

    def display(self, idx, alpha=1):
        img, seg = self[idx]
        img = apply_masks(np.array(img), np.array(seg), alpha=alpha)
        return Image.fromarray(img.astype('uint8'))
