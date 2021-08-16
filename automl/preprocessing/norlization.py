# -*- coding:utf-8 -*-
"""
This is to normalize the data to a normal distribution for some algorithms

@author: Guangqiang.lu
"""
from sklearn.preprocessing import Normalizer
from preprocessing.processing_base import Process


class Normalize(Process):
    def __init__(self):
        super(Normalize, self).__init__()
        self.estimator = Normalizer()


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)

    norm = Normalize()
    norm.fit(x)
    print(norm.transform(x))
    print(norm.name)
