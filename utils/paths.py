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
    path = str(Path(__file__).parent.parent)
    return path


def get_param_config_path():
    """
    this is to get the param config path
    :return: config file path
    """
    path = str(Path(__file__).parent)
    config_path = os.path.join(path,
                               [x for x in os.listdir(path)
                                if x == 'param_config.yml'][0])
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


def load_yaml_file(file_path=None):
    """
    To load yaml file from server path.
    :param file_path: where the file exist
    :return: dictionary
    """
    try:
        if file_path is None:
            # if we don't provide the file_path, try to load root path with file_name: default_algorithms.yaml
            default_file_name = 'default_algorithms.yml'
            file_path = os.path.join(get_root_path(), default_file_name)

        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except yaml.YAMLError as e:
        raise IOError("When try to load yaml file from path: {} get error: {}".format(file_path, e))


def get_file_base_name(path):
    """
    To get base file name based on the path
    :param path:
    :return:
    """
    try:
        name = os.path.basename(path).split('.')[0]
        return name
    except ValueError as e:
        raise ValueError("When to get file path: {} with error: {}".format(path, e))


if __name__ == "__main__":
    print(get_param_config_path())

    print(get_root_path())
