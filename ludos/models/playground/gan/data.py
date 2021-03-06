import json
import os

import numpy as np
import pytorch_lightning as pl
from box import Box
from ludos.utils import viz
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm

ROOT_DIR = os.environ['ROOT_DIR']
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'interim')


def get_transforms(cfg, split='train'):
    if cfg.dm.name == 'cifar10':
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    cfg.dm.transforms.normalize.mean,
                    cfg.dm.transforms.normalize.std)
            ])

    tf = T.Compose(
        [
            T.CenterCrop(480),
            T.Resize(size=(cfg.dm.transforms.size, cfg.dm.transforms.size)),
            T.ToTensor(),
            T.Normalize(
                cfg.dm.transforms.normalize.mean,
                cfg.dm.transforms.normalize.std)
        ])
    return tf


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(CIFAR10DataModule, self).__init__()
        self.batch_size = cfg.dm.params.batch_size
        self.num_workers = cfg.dm.params.num_workers
        self.cfg = cfg

    def prepare_data(self):
        CIFAR10(DATA_PATH)

    def setup(self, *args, **kwargs):
        tf = get_transforms(self.cfg, split='train')
        self.train_set = CIFAR10(DATA_PATH, train=True, transform=tf)

        tf = get_transforms(self.cfg, split='validation')
        self.validation_set = CIFAR10(DATA_PATH, train=False, transform=tf)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            sampler=None,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers)


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(DataModule, self).__init__()
        self.batch_size = cfg.dm.params.batch_size
        self.data_name = cfg.dm.params.data_name
        self.num_workers = cfg.dm.params.num_workers
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage):
        tf = get_transforms(self.cfg, split='train')
        self.train_set = Dataset(self.data_name, split='train', transforms=tf)

        tf = get_transforms(self.cfg, split='validation')
        self.validation_set = Dataset(
            self.data_name, split='validation', transforms=tf)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            sampler=None,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers)


class Dataset(object):
    def __init__(self, data_name, split, transforms=None):
        self.transforms = transforms
        self.data_name = data_name
        self.split = split
        self.ann_file = os.path.join(
            DATA_PATH, data_name, 'annotations_{}.json'.format(split))
        with open(self.ann_file) as f:
            self.dataset = Box(json.load(f))
        self._index()

    def __len__(self):
        return len(self.ids)

    def _index(self):
        self.ids = [d['id'] for d in self.dataset.data]

    def get_img_info(self, idx):
        return self.dataset.data[idx]

    def display(self, idx, alpha=1):
        img = self[idx]
        return Image.fromarray(img.astype('uint8'))

    def __getitem__(self, idx):
        info = self.dataset.data[idx]
        img = Image.open(info['image_path'])
        if self.transforms is not None:
            img = self.transforms(img)
        return img
