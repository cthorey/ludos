import datetime
import os
import uuid

import numpy as np
from box import Box

from hubmap.data import common, reader
from PIL import Image
from tqdm import tqdm


def generate_patches(img, mask, size=256, stride=0.5):
    width = float(size * stride)
    w, h = img.shape[0], img.shape[1]
    u = np.linspace(0, w, int(np.floor(w / width) + 1.0)).astype('int')
    v = np.linspace(0, h, int(np.floor(h / width) + 1.0)).astype('int')
    uu, vv = np.meshgrid(u, v)
    centers = np.vstack((uu.ravel(), vv.ravel())).T
    for c in tqdm(centers, total=len(centers)):
        xmin, xmax = np.maximum(0, c[0] - size // 2), np.minimum(
            w, c[0] + size // 2)
        ymin, ymax = np.maximum(0, c[1] - size // 2), np.minimum(
            h, c[1] + size // 2)
        img_patch = img[ymin:ymax, xmin:xmax]
        if np.sum(np.array(img_patch.shape) == size) != 2:
            continue
        mask_patch = mask[ymin:ymax, xmin:xmax]
        yield img_patch, mask_patch
    return centers


class DatasetConstructor(object):
    def __init__(self, data_name, dest_folder, split, **kwargs):
        self.data_name = data_name
        self.dest_folder = dest_folder
        self.dataset = Box(data=[], info={})
        self.dataset.info.data_created = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M")
        self.dataset.info.update(kwargs)
        self.image_id = 0
        self.split = split

    def save(self, payload, fname, data_type='image'):
        common.dump_payload(payload=payload, fname=fname, data_type=data_type)

    def process_record(self, info, img, mask):
        drecord = Box(info)
        drecord['id'] = self.image_id
        name = uuid.uuid4().hex
        name = 'image_{}.png'.format(name)
        drecord.image_path = os.path.join(self.dest_folder, 'data', name)
        image = Image.fromarray(img)
        self.save(image, drecord.image_path, 'image')

        name = 'seg_{}.png'.format(name)
        drecord.seg_path = os.path.join(self.dest_folder, 'data', name)
        mask = Image.fromarray(mask, 'L')
        self.save(mask, drecord.seg_path, 'image')
        self.dataset.data.append(drecord)
        self.image_id += 1

    def dump_annotations(self):
        """
        Dump the annotations.json to disk.
        """
        fname = 'annotations'
        if self.split is not None:
            fname = 'annotations_{}'.format(self.split)
        fname = os.path.join(self.dest_folder, '{}.json'.format(fname))
        self.save(self.dataset, fname, "annotations")


def process_split(samples, data_name, split, overfit, **kwargs):
    r = reader.Reader()
    dataset = DatasetConstructor(data_name,
                                 split=split,
                                 overfit=overfit,
                                 **kwargs)
    datas = []
    for idx, sample_id in enumerate(samples):
        print('Processing {}/{}'.format(idx, len(samples)))
        img = r.get_image(sample_id)
        mask = r.get_segmentation(sample_id)
        patch_generator = generate_patches(img, mask, size=256, stride=0.5)
        for img_patch, mask_patch in patch_generator:
            dataset.process_record(r[sample_id], img_patch, mask_patch)
            if overfit and dataset.image_id > 5:
                break
        if overfit:
            break
    dataset.dump_annotations()


def get_split(train):
    ids = train.index.to_list()
    return ids[:-3], ids[-3:]


def create_dataset(data_name,
                   overwrite=False,
                   overfit=False,
                   maintainer='clement',
                   total=10000,
                   **kwargs):
    if overfit:
        data_name = '{}_overfit'.format(data_name)
    r = reader.Reader()
    print('-' * 50)
    print('DATASET_NAME: {}'.format(data_name))
    train, validation = get_split(r.train_meta)
    dest_folder = common.init_folder(data_name, overwrite=overwrite)
    entry = Box()
    for split, data in [('train', train), ('validation', validation)]:
        print('Processing {}'.format(split))
        process_split(data,
                      dest_folder=dest_folder,
                      data_name=data_name,
                      split=split,
                      overwrite=overwrite,
                      overfit=overfit)
