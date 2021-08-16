# -*- coding:utf-8 -*-
"""
This is main class that is used for whole processing logic for sklearn.

@author: Guangqiang.lu
"""
from sklearn.base import TransformerMixin

from utils.paths import get_file_base_name


class Process(TransformerMixin):
    """
    This is whole class that is used for pre-processing logic, just give a direction.
    Here for init function that we want to get just class name for later process.
    """
    def __init__(self):
        self.name = self.__class__.__name__
        self.estimator = None

    def fit(self, x, y=None):
        """
        For whole processing logic should provide with label, so that even we don't use
        it, we could just follow sklearn logic.
        :param data:
        :param label:
        :return:
        """
        self.estimator.fit(x, y=y)

    def fit_transform(self, data, y=None):
        """
        Make parent logic just like sklearn.
        :param data:
        :return:
        """
        self.fit(data, y=y)
        return self.transform(data, y=y)

    def transform(self, data, y=None):
        return self.estimator.transform(data)


if __name__ == '__main__':
    p = Process()
    print(p.name)
