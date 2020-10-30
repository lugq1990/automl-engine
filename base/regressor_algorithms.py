# -*- coding:utf-8 -*-
"""
Let's just use whole regression used in sklearn to be instant
here so that we could change or do something change could be easier.

@author: Guangqiang.lu
"""
from auto_ml.hyper_config import (ConfigSpace, UniformHyperparameter, CategoryHyperparameter,
                                                          GridHyperparameter)
from sklearn.base import BaseEstimator


class RegressorClass(BaseEstimator):
    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        pred = self.estimator.predict(x)
        return pred

    def score(self, x, y):
        score = self.estimator.score(x, y)
        return score

    @staticmethod
    def get_search_space():
        raise NotImplementedError


class LinearRegression(RegressorClass):
    def __init__(self, fit_intercept=True,
                 n_jobs=None,
                 normalize=False
                 ):
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs
        self.normalize = normalize
        self.estimator = None

    def fit(self, x, y):
        from sklearn.linear_model import LinearRegression

        self.estimator = LinearRegression(fit_intercept=self.fit_intercept,
                                          n_jobs=self.n_jobs,
                                          normalize=self.normalize)

        self.estimator.fit(x, y)

        return self

    def predict(self, x):
        return super().predict(x)

    def score(self, x, y):
        return super().score(x, y)