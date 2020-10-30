# -*- coding:utf-8 -*-
"""
This is just MinMax logic to process data

@author: Guangqiang.lu
"""
from sklearn.preprocessing import MinMaxScaler
from auto_ml.preprocessing.processing_base import Process


class MinMax(Process):
    def __init__(self):
        super(MinMax, self).__init__()
        self.estimator = MinMaxScaler()

