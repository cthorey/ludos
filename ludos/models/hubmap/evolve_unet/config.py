import copy
import os

from box import Box

C = Box()

# misc options
C.auto_lr_find = True

C.checkpoint = Box()
C.checkpoint.name = '{epoch}-{val_loss:.2f}-{val_dice:.2f}'
C.checkpoint.monitor = 'val_loss'
C.checkpoint.monitor_mode = 'max'

C.early_stopping = Box()
C.early_stopping.min_delta = 0.1
C.early_stopping.patience = 10
C.early_stopping.verbose = True

# trainer
C.trainer = Box()
C.trainer.accumulate_grad_batches = 1

# augmentations
C.augmentation = ''

# models
C.model = Box()
C.model.name = 'unet'
C.model.device = "cuda"
C.model.parameters = Box(dimensions=2,
                         in_channels=4,
                         out_channels=1,
                         channels=(16, 32, 64, 128, 256),
                         strides=(2, 2, 2, 2),
                         num_res_units=2)

# datasets
C.datasets = Box()
# list of the dataset names for training, as present in paths_catalog.py
C.datasets.train = Box(data_name="hubmapd0_overfit", split="train")

# list of the dataset names for testing, as present in paths_catalog.py
C.datasets.test = Box(data_name="hubmapd0_overfit", split="validation")
C.datasets.num_classes = 1

# dataloader
C.dataloader = Box()
C.dataloader.shuffle = True
C.dataloader.num_workers = 4

# solver
C.solver = Box()

C.solver.loss = Box()
C.solver.loss.name = "DiceLoss"
C.solver.loss.params = {"sigmoid": True}

C.solver.optimizer = Box()
C.solver.optimizer.name = "Adam"
C.solver.optimizer.params = {}

C.solver.scheduler = Box()
C.solver.scheduler.name = "OneCycleLR"
C.solver.scheduler.params = Box(total_steps=10000)
C.solver.scheduler_meta = Box(interval="step", monitor="val_loss")
C.solver.ims_per_batch = 32
C.solver.default_lr = 0.0001

# test specific
C.test = Box()
C.test.expected_results = []
C.test.expected_results_sigma_tol = 4
# number of images per batch
# this is global, so if we have 8 gpus and ims_per_batch = 16, each gpu will
# see 2 images per batch
C.test.ims_per_batch = 8


def get():
    return copy.deepcopy(C)
