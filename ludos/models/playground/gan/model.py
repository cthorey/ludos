import os

import numpy as np
import torch
from box import Box
from ludos.models import common
from ludos.models.playground.gan import config, data, network
from ludos.utils import dictionary
from PIL import Image
from torchvision.utils import make_grid


def get_cfg(config_name: str = ""):
    default_cfg = config.get().to_dict()
    flatten_default_cfg = dictionary.flatten(default_cfg)
    cfg = common.get_cfg(
        model_task='playground', model_name='gan', config_name=config_name)
    flatten_cfg = dictionary.flatten(cfg.to_dict())
    flatten_default_cfg.update(flatten_cfg)
    return Box(dictionary.unflatten(flatten_default_cfg))


class Model(common.BaseModel):
    """
    Instance segmentation based on mask-rcnn.
    """
    def __init__(
            self,
            model_type='models',
            model_name='gan',
            model_task='playground',
            model_description="",
            expname=None):
        super(Model, self).__init__(
            model_type=model_type,
            model_name=model_name,
            model_task=model_task,
            model_description=model_description,
            expname=expname)
        if self.expname is not None:
            self.load_network(expname=expname)

    @property
    def normalization(self):
        return self.network.cfg.dm.transforms.normalize

    def build(self, cfg):
        self.network = network.BasicGAN(cfg)
        self.network.setup('train')

    def load_network(self, expname):
        checkpoint_path = os.path.join(
            self.model_folder, '{}_weights.pth'.format(self.expname))
        common.download_weight(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        cfg = checkpoint['extra']['cfg']
        self.network = network.BasicGAN(cfg)
        self.network = self.network.load_from_checkpoint(
            checkpoint_path, is_train=False)
        self.network.eval()

    def generate_sample(self, nrow, use_decoder_as_is=True):
        n = nrow**2
        mu = torch.zeros((n, self.network.cfg.network.latent_dim))
        std = torch.ones((n, self.network.cfg.network.latent_dim))
        dist = torch.distributions.Normal(mu, std)
        z = dist.sample()
        with torch.no_grad():
            preds = self.network.decoder(z)  # Bx3x32x32
            if not use_decoder_as_is:
                scale = torch.exp(self.network.log_p_xz_std)
                dist = torch.distributions.Normal(preds, scale)
                preds = dist.sample()
        mean, std = np.array(self.normalization.mean), np.array(
            self.normalization.std)
        return (
            make_grid(preds, nrow=nrow).permute(1, 2, 0).numpy() * std +
            mean) * 255
