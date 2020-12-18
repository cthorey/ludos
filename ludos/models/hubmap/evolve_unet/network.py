from box import Box

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from ludos.models.hubmap.evolve_unet import architectures, data
from monai import transforms as tf
from monai.networks.nets import UNet
from torch import optim
from torch.utils.data import DataLoader

OPTIMIZER = dict(Adam=optim.Adam, RMSprop=optim.RMSprop)
SCHEDULER = dict(OneCycleLR=optim.lr_scheduler.OneCycleLR)


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
        self.criterion = smp.utils.losses.DiceLoss(activation="sigmoid")
        self.postprocessing = tf.Compose([
            tf.Activations(sigmoid=True),
            tf.AsDiscrete(threshold_values=True)
        ])
        self.metrics = {
            'dice': smp.utils.metrics.Fscore(),
            'iou': smp.utils.metrics.IoU()
        }
        self.net = architectures.build(self.cfg.model.name,
                                       **self.cfg.model.parameters)

    def setup(self, stage):
        tf = data.build_transforms(self.cfg, is_train=True)
        self.train_set = data.TrainingDataset(transforms=tf,
                                              **self.cfg.datasets.train)

        tf = data.build_transforms(self.cfg, is_train=False)
        self.validation_set = data.TrainingDataset(transforms=tf,
                                                   **self.cfg.datasets.test)
        self.test_set = data.TrainingDataset(transforms=tf,
                                             **self.cfg.datasets.test)

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
        return self.net(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
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
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        predictions = self.postprocessing(logits)
        metrics = {'loss': torch.tensor([loss])}
        for key, metric in self.metrics.items():
            metrics[key] = torch.tensor([metric(predictions, targets)])
        return metrics

    def validation_epoch_end(self, outputs):
        logs = dict()
        metrics = outputs[0].keys()
        for metric in metrics:
            arr = [r[metric] for r in outputs]
            logs['val_{}'.format(metric)] = torch.cat(arr).mean()
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
