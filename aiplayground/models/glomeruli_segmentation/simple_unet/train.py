import os

import fire

import pytorch_lightning as pl
from aiplayground.models import experiment
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import trainer


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


def auto_lr_find(network):
    tr = trainer.Trainer(gpus=1)
    result = tr.tuner.lr_find(network)
    return result.suggestion()


def train(config_name, max_epochs=1, maintainer='clement', gpus=1):
    """Main entrypoint to train a model.

    The config file need to lives in training_config/poses_retrieval/rotated_bingham/config_name.yaml.
    The training_config folder is mounted at runtime.

    This will

    1. Generate a new trial for this config -- [config_name]tX where X is increasing interger.
    2. Live reporting on http://ec2-3-8-193-158.eu-west-2.compute.amazonaws.com:8080/dashboard. We use Trains from allegro for the reporting
    3. Train the model
    4. Compute relevant metrics on the validation set and write them back to the modelzoo.
    5. Store the weights/meta_data on s3.

    Later, you can use your model by just calling

    Example:
        .. code-block::

           from aiplayground.models import common
           m = common.load_model(model_task='poses_retrieval',
                                 model_name='rotated_bingham',
                                 expname=[config_name]tX)

    Args:
        config_name (str): name of the config file in training_config folder
        max_epochs (int): Number of epochs to train the model. NOT USED HERE.
        maintainer (str): Name of the maintainer
        gpus (str): List of gpus you want to train seprated by a ,
    """

    cfg = model.get_cfg(config_name)
    task = experiment.LightningExperiment(model_task="poses_retrieval",
                                          model_name="rotated_bingham",
                                          config_name=config_name)
    m = model.Model()
    m.build_network(cfg.to_dict())
    if cfg.solver.scheduler.name == 'OneCycleLR':
        m.network.cfg.solver.scheduler.params.total_steps = len(
            m.network.train_dataloader()) * max_epochs

    if cfg.auto_lr_find:
        lr = auto_lr_find(m.network)
        cfg.solver.default_lr = lr
        m.build_network(cfg.to_dict())
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
    tr.fit(m.network)
    results = tr.test()[0]
    task.upload_checkpoints(checkpoint.best_model_path,
                            expname,
                            cfg=m.network.cfg.to_dict(),
                            **results)
    task.end(expname=expname,
             dataset_name='|'.join(m.network.cfg.datasets.validation),
             score_name="angle_error",
             score=results["mean_val_angle_error"],
             split="validation")


def explore(name):
    advisor = experiment.Bender(model_task='poses_retrieval',
                                model_name='rotated_bingham',
                                exploration_name=name)
    config_name = advisor.suggest()
    train(config_name, **advisor.exploration.common)


if __name__ == '__main__':
    fire.Fire()
