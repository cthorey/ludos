import segmentation_models_pytorch as smp
from ludos.models.hubmap.evolve_unet import unet

ARCHS = {
    'smpunet': smp.Unet,
    'basicunet': unet.BasicUnet,
    'subpixelunet': unet.SubPixelNet,
    'asppunet': unet.ASPPUnet
}


def build(name, **params):
    return ARCHS[name](**params)
