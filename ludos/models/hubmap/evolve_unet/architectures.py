import segmentation_models_pytorch as smp
from ludos.models.hubmap.evolve_unet import unet

ARCHS = {
    'smpmanet': smp.MAnet,
    'smpunetplusplus': smp.UnetPlusPlus,
    'smpunet': smp.Unet,
    'basicunet': unet.BasicUnet,
    'subpixelunet': unet.SubPixelNet,
    'subpixelunetwithFPN': unet.SubPixelNetWithFPN,
    'sesubpixelunetwithFPN': unet.SESubPixelNetWithFPN,
    'asppunet': unet.ASPPUnet
}


def build(name, **params):
    return ARCHS[name](**params)
