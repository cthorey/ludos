import copy
import os

from box import Box

C = Box()

# misc options
C.AUTO_LR_FIND = True

C.CHECKPOINT = Box()
C.CHECKPOINT.NAME = '{epoch}-{val_loss:.2f}-{val_mcc:.2f}'
C.CHECKPOINT.MONITOR = 'val_mcc'
C.CHECKPOINT.MONITOR_MODE = 'max'

C.EARLY_STOPPING = Box()
C.EARLY_STOPPING.min_delta = 0.1
C.EARLY_STOPPING.patience = 10
C.EARLY_STOPPING.verbose = True

# trainer
C.TRAINER = Box()
C.TRAINER.accumulate_grad_batches = 1

# augmentations
C.AUGMENTATION = ''

# models
C.MODEL = Box()
C.MODEL.DEVICE = "cuda"
C.MODEL.UNET = Box(dimensions=2,
                   in_channels=4,
                   out_channels=1,
                   channels=(16, 32, 64, 128, 256),
                   strides=(2, 2, 2, 2),
                   num_res_units=2)

# datasets
C.DATASETS = Box()
# List of the dataset names for training, as present in paths_catalog.py
C.DATASETS.TRAIN = Box(data_name="hubmapd0_overfit", split="train")

# List of the dataset names for testing, as present in paths_catalog.py
C.DATASETS.TEST = Box(data_name="hubmapd0_overfit", split="validation")
C.DATASETS.NUM_CLASSES = 1

# DataLoader
C.DATALOADER = Box()
C.DATALOADER.SHUFFLE = True
C.DATALOADER.NUM_WORKERS = 4

# solver
C.SOLVER = Box()

C.SOLVER.LOSS = Box()
C.SOLVER.LOSS.NAME = "DiceLoss"
C.SOLVER.LOSS.PARAMS = {'sigmoid': True}

C.SOLVER.OPTIMIZER = Box()
C.SOLVER.OPTIMIZER.NAME = 'Adam'
C.SOLVER.OPTIMIZER.PARAMS = {}

C.SOLVER.SCHEDULER = Box()
C.SOLVER.SCHEDULER.NAME = "OneCycleLR"
C.SOLVER.SCHEDULER.PARAMS = Box(total_steps=10000)
C.SOLVER.SCHEDULER_META = Box(interval='step', monitor='val_acc')
C.SOLVER.IMS_PER_BATCH = 32
C.SOLVER.DEFAULT_LR = 0.0001

# test specific
C.TEST = Box()
C.TEST.EXPECTED_RESULTS = []
C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
C.TEST.IMS_PER_BATCH = 8


def get():
    return copy.deepcopy(C)
