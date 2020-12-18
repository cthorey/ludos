import segmentation_models_pytorch as smp

ARCHS = {'unet': smp.Unet}


def build(name, **params):
    return ARCHS[name](**params)
