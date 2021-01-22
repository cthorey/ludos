import os

import numpy as np
from box import Box

import torch
from ludos.models import common
from ludos.models.playground.vaead import config, network
from ludos.utils import dictionary, s3
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision import transforms
from torchvision.utils import make_grid


def get_cfg(config_name: str = ""):
    default_cfg = config.get().to_dict()
    flatten_default_cfg = dictionary.flatten(default_cfg)
    cfg = common.get_cfg(model_task='playground',
                         model_name='vaead',
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
                 model_name='vaead',
                 model_task='playground',
                 model_description="",
                 expname=None):
        super(Model, self).__init__(model_type=model_type,
                                    model_name=model_name,
                                    model_task=model_task,
                                    model_description=model_description,
                                    expname=expname)
        if self.expname is not None:
            self.load_network(expname=expname)

    def build(self, cfg):
        self.network = network.VAE(cfg)
        self.network.setup('train')

    def load_network(self, expname):
        checkpoint_path = os.path.join(self.model_folder,
                                       '{}_weights.pth'.format(self.expname))
        common.download_weight(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        cfg = checkpoint['extra']['cfg']
        self.network = network.VAE(cfg)
        self.network = self.network.load_from_checkpoint(checkpoint_path,
                                                         is_train=False)
        self.network.eval()

    def detect_anomaly(self, img, use_decoder_as_is=True):
        cf10_transforms = transforms.Compose(
            [transforms.ToTensor(),
             cifar10_normalization()])
        x = cf10_transforms(img).unsqueeze(0)
        with torch.no_grad():
            feats = self.network.encoder(x)
            mu, log_var = self.network.fc_mu(feats), self.network.fc_var(feats)
            std = torch.exp(log_var * 0.5)
            z = torch.distributions.Normal(mu, std).sample()
            preds = self.network.decoder(z)
            if not use_decoder_as_is:
                scale = torch.exp(self.network.log_p_xz_std)
                dist = torch.distributions.Normal(preds, scale)
                preds = dist.sample()
        normalize = cifar10_normalization()
        mean, std = np.array(normalize.mean), np.array(normalize.std)
        return ((preds[0].permute(1, 2, 0).numpy() * std + mean) *
                255).astype('uint8')

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
        normalize = cifar10_normalization()
        mean, std = np.array(normalize.mean), np.array(normalize.std)
        return (make_grid(preds, nrow=nrow).permute(1, 2, 0).numpy() * std +
                mean) * 255
