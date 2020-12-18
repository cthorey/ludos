import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.utils import functional
from torch.nn import functional as F

try:
    from segmentation_models_pytorch.base.modules import Activation
except:
    from segmentation_models_pytorch.utils.base import Activation


class Loss(smp.utils.base.Loss):
    def __init__(self, w_bce=1, w_dice=0.5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.w_bce = torch.tensor(w_bce).float()
        self.w_dice = torch.tensor(w_dice).float()

    def dice(self, y_pr, y_gt):
        return 1 - functional.f_score(
            y_pr,
            y_gt,
            beta=1,
            eps=1,
            threshold=None,
            ignore_channels=None,
        )

    def bce(self, y_pr, y_gt):
        return F.binary_cross_entropy(y_pr, y_gt)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return self.w_bce * self.bce(
            y_pr, y_gt.float()) + self.w_dice * self.dice(y_pr, y_gt)
