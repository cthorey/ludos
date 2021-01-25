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
C.network.loss = 'kl'

# dm
C.dm = Box()
C.dm.name = 'cifar10'
C.dm.params = Box()
C.dm.transforms = Box()

# C.dm.params.data_dir = '/workdir/data/interim'
# C.dm.transforms = Box()
# C.dm.transforms.normalize = {
#     'mean': [0.4913725490196078, 0.4823529411764706, 0.4466666666666667],
#     'std': [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]
# }

# C.dm.name = 'custon'
# C.dm.params = Box()
# C.dm.params.data_name = 'lamp'
# C.dm.params.batch_size = 32
# C.dm.transforms = Box()
# C.dm.transforms.resize = {
#     'size': (256,256)
# }
# C.dm.transforms.normalize = {
#     'mean': [0.60482164, 0.59398557, 0.56691167],
#     'std': [0.23168588, 0.24707217, 0.24729942]
# }


def get():
    return copy.deepcopy(C)
