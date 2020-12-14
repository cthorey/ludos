from box import Box

import pytorch_lightning as pl
import torch
from ludos.data.hubmap.data import PatchDataset
from monai import data, losses, metrics
from monai import transforms as tf
from monai.networks.nets import UNet
from torch import optim
from torch.utils.data import DataLoader

OPTIMIZER = dict(Adam=optim.Adam, RMSprop=optim.RMSprop)
SCHEDULER = dict(OneCycleLR=optim.lr_scheduler.OneCycleLR)
LOSSES = dict(DiceLoss=losses.DiceLoss)


class LightningUNet(pl.LightningModule):
    def __init__(self, cfg, is_train=True):
        super().__init__()
        if isinstance(cfg, Box):
            raise ValueError('Pass a dict instead')
        self.save_hyperparameters('cfg', 'is_train')
        self.cfg = Box(cfg)
        self.batch_size = self.cfg.solver.ims_per_batch
        self.learning_rate = self.cfg.solver.default_lr
        self.cfg = Box(cfg)
        self.unet = UNet(**self.cfg.model.parameters)
        self.criterion = LOSSES[self.cfg.solver.loss.name](
            **self.cfg.solver.loss.get('params', {}))
        self.augmentation = None
        if cfg.get('augmentation', ''):
            self.augmentation = augs.augmentations[cfg.get('augmentation', '')]
        self.postprocessing = tf.Compose([
            tf.Activations(sigmoid=True),
            tf.AsDiscrete(threshold_values=True)
        ])
        self.metric = metrics.DiceMetric(sigmoid=False, reduction='none')

    def setup(self, stage):
        train_ds = PatchDataset(**self.cfg.datasets.train)
        train_transforms = tf.Compose([
            tf.LoadImaged(keys=["img", "seg"]),
            tf.AddChanneld(keys=["seg"]),
            tf.AsChannelFirstd(keys=["img"]),
            tf.ScaleIntensityd(keys="img"),
            tf.ToTensord(keys=["img", "seg"]),
        ])
        self.train_set = data.Dataset(train_ds.sequence,
                                      transform=train_transforms)

        val_ds = PatchDataset(**self.cfg.datasets.test)
        val_transforms = tf.Compose([
            tf.LoadImaged(keys=["img", "seg"]),
            tf.AddChanneld(keys=["seg"]),
            tf.AsChannelFirstd(keys=["img"]),
            tf.ScaleIntensityd(keys="img"),
            tf.ToTensord(keys=["img", "seg"]),
        ])
        self.validation_set = data.Dataset(val_ds.sequence,
                                           transform=val_transforms)
        self.test_set = data.Dataset(val_ds.sequence, transform=val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          sampler=None,
                          num_workers=self.cfg.dataloader.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_set,
                          batch_size=self.batch_size,
                          num_workers=self.cfg.dataloader.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.cfg.dataloader.num_workers)

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch['img'], batch['seg']
        preds = self(images)
        loss = self.criterion(preds, targets)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch['img'], batch['seg']
        logits = self(images)
        loss = self.criterion(logits, targets)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        predictions = self.postprocessing(logits)
        dices = self.metric(predictions, targets)
        return {'dices': dices, 'val_loss': loss}

    def validation_epoch_end(self, outputs):
        logs = dict()
        logs['val_dice'] = torch.cat([r['dices'] for r in outputs]).mean()
        logs['val_loss'] = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log_dict(logs, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        params = self.cfg.solver.optimizer.get('params', {})
        params['lr'] = self.learning_rate
        optimizer = OPTIMIZER[self.cfg.solver.optimizer.name](
            self.parameters(), **dict(params))
        params = self.cfg.solver.scheduler.get('params', {})
        if self.cfg.solver.scheduler.name == 'OneCycleLR':
            params['max_lr'] = self.learning_rate
        scheduler_lr = SCHEDULER[self.cfg.solver.scheduler.name](
            optimizer, **dict(params))
        scheduler = self.cfg.solver.scheduler_meta
        scheduler['scheduler'] = scheduler_lr

        return [optimizer], [scheduler]
