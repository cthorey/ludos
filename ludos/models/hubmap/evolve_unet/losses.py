import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import losses
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.utils import functional
from segmentation_models_pytorch.utils.losses import DiceLoss
from torch import nn
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self, bce=1.0, dice=0.5, lovasz=0.0, soft_bce=True):
        super().__init__()
        if soft_bce:
            self.bce = losses.soft_bce.SoftBCEWithLogitsLoss(smooth_factor=0.1)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(activation=None)
        self.lovasz = losses.lovasz.LovaszLoss(mode='binary',
                                               from_logits=False)
        self.activation = Activation('sigmoid')
        self.w_bce = bce
        self.w_dice = dice
        self.w_lovasz = lovasz

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        loss = self.w_bce * self.bce(y_pr, y_gt.float())
        loss = loss + self.w_dice * self.dice(y_pr, y_gt)
        loss = loss + self.w_lovasz * self.lovasz(y_pr, y_gt)
        return loss
