import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import losses
from segmentation_models_pytorch.utils import functional
from torch import nn
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self, bce=1.0, dice=0.5, lovasz=0.0):
        super().__init__()
        self.soft_bce = losses.soft_bce.SoftBCEWithLogitsLoss()
        self.dice = losses.dice.DiceLoss('binary', from_logits=False)
        self.lovasz = losses.lovasz.LovaszLoss(mode='binary',
                                               from_logits=False)
        self.w_bce = bce
        self.w_dice = dice
        self.w_lovasz = lovasz

    def forward(self, y_pr, y_gt):
        y_pr = F.logsigmoid(y_pr).exp()
        loss = 0.0
        if self.w_bce != 0:
            loss = loss + self.w_bce * self.soft_bce(y_pr, y_gt.float())
        if self.w_dice != 0:
            loss = loss + self.w_dice * self.dice(y_pr, y_gt)
        if self.w_lovasz != 0:
            loss = loss + self.w_lovasz * self.lovasz(y_pr, y_gt)
        return loss
