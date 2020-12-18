import json
import os

import numpy as np
from box import Box

import pandas as pd
from monai.data import image_reader
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT_DIR = os.environ['ROOT_DIR']
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'hubmap-kidney-segmentation')


def rle2mask(mask_rle, shape=(1600, 256)):
    '''
    Args:
        mask_rle: run-length as string formated (start length)
        shape: (width,height) of array to return

    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


class Reader(object):
    def __init__(self, path=DATA_PATH):
        self.data_path = path
        self.train_meta = pd.read_csv(os.path.join(
            DATA_PATH, "train.csv")).set_index('id')
        info = pd.read_csv(
            os.path.join(DATA_PATH, "HuBMAP-20-dataset_information.csv"))
        info = info.to_dict(orient='records')
        self.info = {k['image_file'].split('.')[0]: k for k in info}

    @property
    def ids(self):
        return self.train_meta.index.tolist()

    def __getitem__(self, sample_id):
        return self.info[sample_id]

    def get_image(self, sample_id):
        reader = image_reader.ITKReader()
        raw_image = reader.read(
            os.path.join(DATA_PATH, 'train', "{}.tiff".format(sample_id)))
        image, _ = reader.get_data(raw_image)
        return image

    def get_segmentation_from_rle(self, sample_id, s):
        rle = self.train_meta.loc[sample_id].encoding
        return rle2mask(rle, shape=s)

    def get_segmentation(self, sample_id, debug=False):
        fname = os.path.join(DATA_PATH, 'train', "{}.json".format(sample_id))
        with open(fname, 'r') as f:
            data = json.load(f)
        mask = None
        for idx, globul in tqdm(enumerate(data), total=len(data)):
            height, width = self[sample_id]['height_pixels'], self[sample_id][
                'width_pixels']
            img = Image.new('L', (width, height), 0)
            for polygon in globul['geometry']['coordinates']:
                p = [tuple(xy) for xy in polygon]
                ImageDraw.Draw(img).polygon(p, outline=1, fill=1)
            if mask is None:
                mask = np.array(img).astype(np.bool)
                continue
            mask += np.array(img).astype(np.bool)
            if debug:
                break
        return mask
