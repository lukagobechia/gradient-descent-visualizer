import os

import yaml


def load_config(path=None):
    if path is None:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        path = os.path.normpath(os.path.join(root, "config.yaml"))

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
