import copy
import os

from box import Box

C = Box()

C.checkpoint = Box()
C.checkpoint.name = '{epoch}-{loss:.2f}'
C.checkpoint.monitor = 'loss'
C.checkpoint.monitor_mode = 'min'

C.early_stopping = Box()
C.early_stopping.min_delta = 0.1
C.early_stopping.patience = 10
C.early_stopping.verbose = True

# trainer
C.trainer = Box()
C.trainer.accumulate_grad_batches = 1

# solver
C.network = Box()
C.network.lr = 0.0001
C.network.enc_out_dim = 512
C.network.latent_dim = 256
C.network.input_height = 32


def get():
    return copy.deepcopy(C)
