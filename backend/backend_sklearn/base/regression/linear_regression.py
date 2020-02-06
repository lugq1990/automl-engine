# -*- coding:utf-8 -*-
"""
This is to create a Linear regression object that we
could use for creating a object as base estimator.

@author: Guangqiang.lu
"""
from sklearn.linear_model import LinearRegression


class LR():
    def __init__(self, fit_intercept=True,
                 n_jobs=None,
                 normalize=False
                 ):
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs
        self.normalize = normalize
        self.estimator = None

    def fit(self, x, y):
        self.estimator = LinearRegression(fit_intercept=self.fit_intercept,
                                          n_jobs=self.n_jobs,
                                          normalize=self.normalize)

        self.estimator.fit(x, y)

        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def score(self, x, y):
        return self.estimator.score(x, y)

