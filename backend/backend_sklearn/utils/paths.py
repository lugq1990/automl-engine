# -*- coding:utf-8 -*-
"""
This is whole path functionality that we could use
in this project.

@author: Guangqiang.lu
"""
import os
from pathlib import Path
import yaml


def get_root_path():
    """
    this is to get the root path of the code
    :return: path
    """
    path = str(Path(__file__).parent.parent.parent.parent)
    return path


def get_param_config_path():
    """
    this is to get the param config path
    :return: config file path
    """
    path = str(Path(__file__).parent)
    config_path = os.path.join(path,
                               [x for x in os.listdir(path)
                                if x == 'param_config.yaml'][0])
    return config_path


def load_param_config():
    """
    this is to load yaml config object.
    :return: dictionary config
    """
    try:
        with open(get_param_config_path(), 'r') as f:
            config = yaml.safe_load(f)

        return config
    except yaml.YAMLError as e:
        raise IOError("When try to read config file with error: %s" % e)


if __name__ == "__main__":
    print(get_param_config_path())

    print(load_param_config())
