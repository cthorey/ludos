import os

from box import Box

import torch
from ludos.models import common
from ludos.models.playground.vae import config, network
from ludos.utils import dictionary, s3


def get_cfg(config_name: str = ""):
    default_cfg = config.get().to_dict()
    flatten_default_cfg = dictionary.flatten(default_cfg)
    cfg = common.get_cfg(model_task='playground',
                         model_name='vae',
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
                 model_name='vae',
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

    def generate_sample(self):
        pass
