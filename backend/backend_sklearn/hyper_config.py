# -*- coding:utf-8 -*-
"""
Use a class get generate the hyper-parameters

@author: Guangqiang.lu
"""
import numpy as np
from collections import defaultdict


class ConfigSpace(object):
    def __init__(self):
        self.hyper_space = defaultdict(list)

    def add_hyper(self, hyper_list):
        if not isinstance(hyper_list, list):
            hyper_list = [hyper_list]

        for hyper in hyper_list:
            hyper_name = hyper.name
            for value in hyper.get_values():
                self.hyper_space[hyper_name].append(value)
        return self

    def get_hypers(self):
        for key, values in self.hyper_space.items():
            print("key: ", key)
            print("Values: ", values)
            # print("Get hyper key: {}, {}".format(key, '\t'.join(str(values))))


class Hyperarameter(object):
    """
    Whole hyperparameter parent class that use random logic
    """
    def __init__(self, name):
        self.name = name

    def get_values(self):
        raise NotImplementedError


class NormalHyperameter(Hyperarameter):
    def __init__(self, name, loc=0, scale=1, size=5):
        super().__init__(name=name)
        self.name = name
        self.loc = loc if loc > 0 else 0
        self.scale = scale if scale > 0 else 1
        self.size = size

    def get_values(self):
        """
        get sorted sampled values list
        :return: sorted list
        """
        samples = np.random.normal(loc=self.mean, scale=self.std, size=self.n_samples).tolist()
        return sorted(samples)


class UniformHyperparameter(Hyperarameter):
    def __init__(self, name, low=0, high=100, size=5):
        super().__init__(name=name)
        self.name = name
        self.low = low if low > 0 else 0
        self.high = high if high > 0 else 1
        self.size = size

    def get_values(self):
        """
        get uniform samples with uniform distribution
        :return:
        """
        samples = np.random.uniform(low=self.low, high=self.high, size=self.size).tolist()
        return sorted(samples)


class CategoryHyperparameter(Hyperarameter):
    def __init__(self, name, categories=[], default=None):
        super().__init__(name=name)
        self.name = name
        self.categories = categories
        self.default = default

    def get_values(self):
        """
        if and only if there are too many choices that need to
        choose, otherwise just return the whole categories
        if default is provided, then just use the default value.
        :return:
        """
        if self.default is not None:
            samples = [self.default]
        else:
            samples = self.categories

        return samples


class GridHyperparameter(Hyperarameter):
    """
    Use grid search to find whole wanted values
    """
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def get_values(self):
        return self.values

