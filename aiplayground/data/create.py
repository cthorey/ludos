import importlib
import time

import fire

from aiplayground.data import round0

MAPPING = dict(round0=round0.create_dataset)


def create_dataset(data_name,
                   data_type,
                   overfit=True,
                   size=1024,
                   stride=0.5,
                   overwrite=True):
    try:
        MAPPING[data_type](data_name=data_name,
                           overwrite=overwrite,
                           overfit=overfit,
                           size=size,
                           stride=stride)
    except Exception as e:
        print(e)
        time.sleep(1000000)


if __name__ == '__main__':
    fire.Fire()
