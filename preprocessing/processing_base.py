# -*- coding:utf-8 -*-
"""
This is main class that is used for whole processing logic for sklearn.

@author: Guangqiang.lu
"""
from auto_ml.utils.paths import get_file_base_name


class Process(object):
    """
    This is whole class that is used for pre-processing logic, just give a direction.
    Here for init function that we want to get just class name for later process.
    """
    def __init__(self):
        self.name = self.__class__.__name__
        self.estimator = None

    def fit(self, data):
        self.estimator.fit(data)

    def fit_transform(self, data):
        """
        Make parent logic just like sklearn.
        :param data:
        :return:
        """
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        return self.estimator.transform(data)


if __name__ == '__main__':
    p = Process()
    print(p.name)