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

    def get_segmentation(self, sample_id):
        fname = os.path.join(DATA_PATH, 'train', "{}.json".format(sample_id))
        with open(fname, 'r') as f:
            data = json.load(f)
        mask = None
        for globul in tqdm(data):
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
        return mask
