# -*- coding:utf-8 -*-
"""
This is base estimator that we could use to create
a Logistic Regression model, in fact we do some
hyper-parameters setting for this model

@author: Guangqiang.lu
"""
import numpy as np
from auto_ml.backend.backend_sklearn.hyper_config import (ConfigSpace, UniformHyperparameter, CategoryHyperparameter,
                                                          GridHyperparameter)


class LogisticRegression():
    def __init__(self, C=1.,
                 class_weight=None,
                 dual=False,
                 fit_intercept=True,
                 penalty='l2',
                 n_jobs=None,
                 random_state=1234
                 ):
        self.C = C
        self.class_weight = class_weight
        self.dual = dual
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, x, y):
        from sklearn.linear_model import LogisticRegression

        self.estimator = LogisticRegression(C=self.C,
                                class_weight=self.class_weight,
                                dual=self.dual,
                                fit_intercept=self.fit_intercept,
                                penalty=self.penalty,
                                n_jobs=self.n_jobs,
                                random_state=self.random_state)

        self.estimator.fit(x, y)
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def score(self, x, y):
        return self.estimator.score(x, y)

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        c_list = UniformHyperparameter(name="C", low=0.1, high=10, size=3)
        dual = CategoryHyperparameter(name="dual", categories=[True, False])
        grid = GridHyperparameter(name="C", values=[1, 2, 3])

        config.add_hyper([c_list, dual, grid])

        # config.get_hypers()
        return config

if __name__ == '__main__':
    lr = LogisticRegression()

    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)

    lr.fit(x, y)

    print(lr.score(x, y))
    lr.get_search_space()

