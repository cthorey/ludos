import os

import fire

import pytorch_lightning as pl
from ludos.models import experiment
from ludos.models.playground.vae import data, model
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import trainer

DM = {'cifar10': data.CCIFAR10DataModule, 'custom': data.DataModule}


def get_callbacks(cfg, output_dir):
    cbacks = []
    checkpoint_path = os.path.join(output_dir, cfg.checkpoint.name)
    checkpoint = pl_callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_last=False,
                                              monitor=cfg.checkpoint.monitor,
                                              mode=cfg.checkpoint.monitor_mode)
    cs = [
        pl.callbacks.EarlyStopping(monitor=cfg.checkpoint.monitor,
                                   mode=cfg.checkpoint.monitor_mode,
                                   **cfg.early_stopping),
        pl.callbacks.LearningRateMonitor(),
    ]
    return checkpoint, cs


def train(config_name, max_epochs=1, maintainer='clement', gpus=1):
    """Main entrypoint to train a model.
    """

    cfg = model.get_cfg(config_name)
    m = model.Model()
    m.build(cfg.to_dict())
    task = experiment.LightningExperiment(model_task=m.model_task,
                                          model_name=m.model_name,
                                          config_name=config_name)

    expname = task.next_trial_name()
    logger = task.start(expname)
    logger.connect(cfg)
    output_dir = os.path.join(m.model_folder, expname)
    checkpoint, cbacks = get_callbacks(cfg, output_dir)
    tr = trainer.Trainer(gpus=gpus,
                         default_root_dir=output_dir,
                         max_epochs=max_epochs,
                         checkpoint_callback=checkpoint,
                         callbacks=cbacks,
                         **cfg.trainer)
    tr.trains_task = logger
    dm = DM[cfg.dm.name](cfg)
    tr.fit(m.network, dm)
    task.upload_checkpoints(checkpoint.best_model_path,
                            expname,
                            cfg=m.network.cfg.to_dict())


if __name__ == '__main__':
    fire.Fire()
