#!/usr/bin/env python3

import os

import fire
import yaml

from ludos.models.serving import LudosServer

ROOT_DIR = os.environ['ROOT_DIR']


def get_registry():
    registry_path = os.path.join(ROOT_DIR, 'registry.yaml')
    if not os.path.isfile(registry_path):
        return dict()
    with open(registry_path, 'r') as f:
        registry = yaml.load(f)
    return registry


def spawn_server(host, port):
    """
    Spawn the ludos server

    Args:
        host (str): IP of the host
        port (int): Port you want to exposed to the world.
    """
    server = LudosServer(host=host, port=port)
    registry = get_registry()
    for key, model in registry.items():
        server.load_model(model_id=key, **model)
    server.start()


if __name__ == '__main__':
    fire.Fire(spawn_server)
