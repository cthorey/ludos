import os

import numpy as np
from box import Box

import torch
import torch.nn as nn
from ludos.models import common
from ludos.models.hubmap.evolve_unet import config, data, network
from ludos.utils import dictionary, s3
from PIL import Image


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_cfg(config_name: str = ""):
    default_cfg = config.get().to_dict()
    flatten_default_cfg = dictionary.flatten(default_cfg)
    cfg = common.get_cfg(model_task='hubmap',
                         model_name='evolve_unet',
                         config_name=config_name)
    flatten_cfg = dictionary.flatten(cfg.to_dict())
    flatten_default_cfg.update(flatten_cfg)
    return Box(dictionary.unflatten(flatten_default_cfg))


class Model(common.BaseModel):
    """
    Instance segmentation based on mask-rcnn.
    """
    def __init__(self,
                 model_type='models',
                 model_name='evolve_unet',
                 model_task='hubmap',
                 model_description="",
                 expname=None):
        super(Model, self).__init__(model_type=model_type,
                                    model_name=model_name,
                                    model_task=model_task,
                                    model_description=model_description,
                                    expname=expname)
        cfg = config.get().to_dict()
        self.build_network(cfg, expname=expname)

    def build_network(self, cfg, expname=None):
        arch = cfg['model']['name']
        self.network = network.LightningUNet(cfg, is_train=expname is None)
        if self.expname is not None:
            checkpoint_path = os.path.join(
                self.model_folder, '{}_weights.pth'.format(self.expname))
            if not os.path.isfile(checkpoint_path):
                s3.download_from_bucket(self.bucket, checkpoint_path)
            print('Reloading from {}'.format(checkpoint_path))
            self.network = self.network.load_from_checkpoint(checkpoint_path,
                                                             is_train=False)
            self.network.eval()

    def prepare_inputs(self, images, device):
        transforms = data.build_transforms(self.network.cfg, is_train=False)
        imgs = images if isinstance(images, list) else [images]
        batch = torch.stack([
            transforms(image=np.array(img)[:, :, :3])['image'].to(device)
            for img in imgs
        ], 0)
        return batch

    def predict(self, images, device='cuda'):
        """
        Args:
            images (dict): dict(footprint:np.array,image: np.array)
        """
        device = torch.device(device)
        self.network.to(device)
        batch = self.prepare_inputs(images, device)
        with torch.no_grad():
            logits = self.network(batch)
        return self.network.postprocessing(logits)

    def predict_from_batch(self, batch, device='cuda'):
        device = torch.device(device)
        self.network.to(device)
        with torch.no_grad():
            logits = self.network(batch.to(device))
        return self.network.postprocessing(logits)

    def predict_with_tta_from_batch(self, batch, tf, device='cuda'):
        device = torch.device(device)
        self.network.to(device)
        results = []
        for transformer in tf:
            abatch = transformer.augment_image(batch.to(device))
            with torch.no_grad():
                logits = self.network(abatch)
            logits = transformer.deaugment_mask(logits)
            results.append(logits)
        logits = torch.stack(results).mean(0)
        return self.network.postprocessing(logits)
