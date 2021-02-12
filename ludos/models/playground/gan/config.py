import copy
import os

from box import Box

C = Box()

C.checkpoint = Box()
C.checkpoint.name = '{epoch}-{loss:.2f}'
C.checkpoint.monitor = 'disc_loss'
C.checkpoint.monitor_mode = 'min'

C.early_stopping = Box()
C.early_stopping.min_delta = 0.1
C.early_stopping.patience = 10
C.early_stopping.verbose = True

# trainer
C.trainer = Box()
C.trainer.accumulate_grad_batches = 1

# imput
C.input = Box()
C.input.shape = (3, 224, 224)

# solver
C.network = Box()
C.network.generator = Box()
C.network.generator.lr = 0.0001
C.network.generator.betas = (0.5, 0.999)
C.network.discriminator = Box()
C.network.discriminator.lr = 0.0001
C.network.discriminator.betas = (0.5, 0.999)
C.network.latent_dim = 512

# dm
C.dm = Box()
C.dm.name = 'cifar10'
C.dm.params = Box()
C.dm.transforms = Box()


def get():
    return copy.deepcopy(C)
