# -*- coding:utf-8 -*-
"""
To process data with standard logic

@author: Guangqiang.lu
"""
from sklearn.preprocessing import StandardScaler
from preprocessing.processing_base import Process


class Standard(Process):
    def __init__(self):
        super(Standard, self).__init__()
        self.estimator = StandardScaler()


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)

    s = Standard()
    s.fit(x, y)
    print(s.transform(x))
    print(s.fit_transform(x))
