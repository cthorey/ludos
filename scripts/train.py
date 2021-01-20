#!/usr/bin/env python2
import os
import sys
import uuid

import docker
import fire

VOLUMES = [
    '~/.aws:/root/.aws:rw', '~/workdir/ludos/.pgpass:/root/.pgpass:rw',
    '~/workdir/training_config:/workdir/training_config:rw',
    '/mnt/hdd/models/cache:/root/.cache:rw', '~/workdir/ludos:/workdir:rw',
    '~/.clearml.conf:/root/clearml.conf:rw', '/mnt/hdd/data:/workdir/data:rw',
    '/mnt/hdd/models:/workdir/models:rw'
]
ROOT_DIR = os.path.join(os.environ['HOME'], 'workdir', 'omatai')
docker_client = docker.from_env()
IMAGE = 'ludos:{}'


def create_dataset(model_task,
                   dataset_type,
                   data_name='auto',
                   overwrite=True,
                   overfit=False,
                   image_tag='latest',
                   dry_run=False,
                   **kwargs):
    sys.path.append('./scripts')
    import docker_utils
    cmd = docker_utils.parse_cmd(script="/workdir/ludos/data/constructor.py",
                                 fn="create_dataset",
                                 model_task=model_task,
                                 dataset_type=dataset_type,
                                 data_name=data_name,
                                 overwrite=overwrite,
                                 overfit=overfit,
                                 **kwargs)
    if dry_run:
        print(cmd)
        cmd = "sleep infinity"
    envs = docker_utils.create_envs()
    volumes = docker_utils.create_volumes(VOLUMES)
    name = '{}_{}'.format(model_task, uuid.uuid4().hex)
    docker_client.containers.run(image=IMAGE.format(image_tag),
                                 name=name,
                                 runtime='nvidia',
                                 command=cmd,
                                 detach=True,
                                 stdout=True,
                                 environment=envs,
                                 remove=True,
                                 volumes=volumes,
                                 shm_size="32G")


def train(config_name,
          model_task,
          model_name,
          image_tag='latest',
          devices=0,
          max_epochs=30,
          maintainer='clement',
          detach=True,
          dry_run=False):
    sys.path.append('./scripts')
    import docker_utils
    cmd = docker_utils.parse_cmd(
        script="/workdir/ludos/models/{}/{}/train.py".format(
            model_task, model_name),
        fn='train',
        config_name=config_name,
        max_epochs=max_epochs,
        maintainer=maintainer,
        gpus=1)
    if dry_run:
        print(cmd)
        cmd = "sleep infinity"
    envs = docker_utils.create_envs(devices)
    volumes = docker_utils.create_volumes(VOLUMES)
    name = '{}_{}'.format(config_name, uuid.uuid4().hex)
    remove = False if not detach else True
    docker_client.containers.run(image=IMAGE.format(image_tag),
                                 name=name,
                                 runtime='nvidia',
                                 command=cmd,
                                 detach=detach,
                                 stdout=True,
                                 environment=envs,
                                 remove=remove,
                                 volumes=volumes,
                                 shm_size="32G")


def explore(model_task,
            model_name,
            name,
            image_tag='latest',
            devices=0,
            detach=True,
            dry_run=False):
    sys.path.append('./scripts')
    import docker_utils
    cmd = docker_utils.parse_cmd(
        script="/workdir/ludos/models/{}/{}/train.py".format(
            model_task, model_name),
        fn='explore',
        name=name)
    if dry_run:
        print(cmd)
        cmd = 'sleep infinity'
    envs = docker_utils.create_envs(devices)
    volumes = docker_utils.create_volumes(VOLUMES)
    name = '{}_{}'.format(name, uuid.uuid4().hex)
    remove = False if not detach else True
    policy = docker_utils.create_policy("unless-stopped")
    docker_client.containers.run(image=IMAGE.format(image_tag),
                                 name=name,
                                 restart_policy=policy,
                                 runtime='nvidia',
                                 command=cmd,
                                 detach=detach,
                                 stdout=True,
                                 environment=envs,
                                 volumes=volumes,
                                 shm_size="32G")


if __name__ == '__main__':
    fire.Fire()
