# -*- coding:utf-8 -*-
"""
This is to create a Ensemble object that we could use
to do model construction.

@author: Guangqiang.lu
"""
from sklearn.base import BaseEstimator


class Ensemble(BaseEstimator):
    def __init__(self, model_name_list=None):
        self.model_name_list = model_name_list


