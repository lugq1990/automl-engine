# -*- coding:utf-8 -*-
"""
Let's just use whole classifier used in sklearn to be instant
here so that we could change or do something change could be easier.

@author: Guangqiang.lu
"""
from auto_ml.backend.backend_sklearn.hyper_config import (ConfigSpace, UniformHyperparameter, CategoryHyperparameter,
                                                          GridHyperparameter)
from sklearn.base import BaseEstimator


class ClassifierClass(BaseEstimator):
    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        pred = self.estimator.predict(x)
        return pred

    def score(self, x, y):
        score = self.estimator.score(x, y)
        return score

    def predict_proba(self, x):
        try:
            prob = self.estimator.predict_proba(x)
            return prob
        except:
            raise NotImplementedError("Current estimator doesn't support predict_proba!")

    @staticmethod
    def get_search_space():
        raise NotImplementedError


class LogisticRegression(ClassifierClass):
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
        return super().predict(x)

    def score(self, x, y):
        return super().score(x, y)

    def predict_proba(self, x):
        return super().predict_proba(x)

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        c_list = UniformHyperparameter(name="C", low=0.1, high=10, size=3)
        dual = CategoryHyperparameter(name="dual", categories=[True, False])
        grid = GridHyperparameter(name="C", values=[1, 2, 3])

        config.add_hyper([c_list, dual, grid])

        # config.get_hypers()
        return config


class SupportVectorMachine(ClassifierClass):
    def __init__(self, C=1.,
                 class_weight=None,
                 kernel='rbf',
                 probability=True,
                 random_state=1234
                 ):
        self.C = C
        self.class_weight = class_weight
        self.kernel = kernel
        self.probability = probability
        self.random_state = random_state

    def fit(self, x, y):
        from sklearn.svm import SVC

        self.estimator = SVC(C=self.C,
                             class_weight=self.class_weight,
                             kernel=self.kernel,
                             probability=self.probability,
                             random_state=self.random_state
                            )

        self.estimator.fit(x, y)
        return self

    def predict(self, x):
        return super().predict(x)

    def score(self, x, y):
        return super().score(x, y)

    def predict_proba(self, x):
        return super().predict_proba(x)

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
    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)

    lr = LogisticRegression()

    lr.fit(x, y)
    print(lr.score(x, y))
    print(lr.predict_proba(x))
    print(lr.predict(x))
