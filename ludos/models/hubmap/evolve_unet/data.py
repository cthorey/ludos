import copy
import os

import numpy as np

import albumentations as albu
from albumentations.pytorch import ToTensorV2
from ludos.data.hubmap import data
from ludos.utils import viz
from PIL import Image
from torchvision import transforms as T

ROOT_DIR = os.environ['ROOT_DIR']


def build_transforms(cfg, is_train, debug=False):
    to_compose = [albu.Resize(*cfg.inputs.size)]
    if is_train and cfg.augmentation.enable:
        to_compose.extend([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.Transpose(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.2,
                                  rotate_limit=0,
                                  shift_limit=0.2,
                                  p=0.2,
                                  border_mode=0),
            albu.IAAAdditiveGaussianNoise(p=0.2),
            albu.IAAPerspective(p=0.5),
            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightness(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),
            albu.OneOf(
                [
                    albu.IAASharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),
            albu.OneOf(
                [
                    albu.RandomContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
            albu.Compose(
                [albu.VerticalFlip(p=0.5),
                 albu.RandomRotate90(p=0.5)])
        ])
    if debug:
        return albu.Compose(to_compose)
    to_compose.append(albu.Normalize(**cfg.inputs.normalize))
    to_compose.append(ToTensorV2(transpose_mask=True))
    return albu.Compose(to_compose)


class TrainingDataset(data.PatchDataset):
    def __init__(self, data_name, split, transforms):
        super(TrainingDataset, self).__init__(data_name, split)
        self.transforms = transforms

    def display(self, idx, alpha=1):
        img, seg = self[idx]
        img = viz.apply_masks(img, seg, alpha=alpha)
        return Image.fromarray(img.astype('uint8'))

    def __getitem__(self, idx):
        info = self.dataset.data[idx]
        img = np.array(Image.open(info['image_path']))[:, :, :3]
        mask = np.expand_dims(np.array(Image.open(info['seg_path'])), -1)
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        return img, mask
