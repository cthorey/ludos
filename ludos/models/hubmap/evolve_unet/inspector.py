import random

import numpy as np

from ludos.utils import viz
from PIL import Image


def inspect_one(m, ds, idx=None):
    """ Inspect the perf of a model

    .. code-block:: python
          tf = data.build_transforms(m.network.cfg,is_train=False,debug=True)
          ds =data.TrainingDataset('hubmapd1',split='train',transforms=tf)4

    """
    if idx is None:
        idx = random.choice(ds.ids)
    img, mask = ds[idx]
    pred_mask = m.predict(
        images=Image.fromarray(img)).to('cpu').numpy()[0].transpose(1, 2, 0)
    pred_composite = Image.fromarray(viz.apply_masks(img, pred_mask))
    composite = Image.fromarray(viz.apply_masks(img, mask))
    arr = np.hstack((composite, pred_composite))
    return Image.fromarray(arr)


def inspect(m, ds, visd, n=50):
    for _ in range(n):
        img = inspect_one(m, ds)
        img = np.array(img).transpose(2, 0, 1)
        visd.image(img,
                   env="{}/{}/{}".format(ds.data_name, ds.split, m.expname))
