# -*- coding:utf-8 -*-
"""
Let's just use whole classifier used in sklearn to be instant
here so that we could change or do something change could be easier.

@author: Guangqiang.lu
"""
import warnings
from sklearn.base import BaseEstimator

from auto_ml.hyper_config import (ConfigSpace, UniformHyperparameter, CategoryHyperparameter,
                                                          GridHyperparameter)

warnings.simplefilter('ignore')


class ClassifierClass(BaseEstimator):
    def __init__(self):
        super(ClassifierClass, self).__init__()
        self.name = self.__class__.__name__

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
        """
        This is to get predefined search space for different algorithms,
        and we should use this to do cross validation to get best fitted
        parameters.
        :return:
        """
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
        super(LogisticRegression, self).__init__()
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
        # dual = CategoryHyperparameter(name="dual", categories=[True, False])
        grid = GridHyperparameter(name="C", values=[1, 2, 3])

        config.add_hyper([c_list, grid])

        # config.get_hypers()
        return config.get_hypers()


class SupportVectorMachine(ClassifierClass):
    def __init__(self, C=1.,
                 class_weight=None,
                 kernel='rbf',
                 probability=True,
                 random_state=1234
                 ):
        super(SupportVectorMachine, self).__init__()
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
        # dual = CategoryHyperparameter(name="dual", categories=[True, False])
        grid = GridHyperparameter(name="C", values=[10, 20, 30])

        config.add_hyper([c_list, grid])

        return config.get_hypers()


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import GridSearchCV

    x, y = load_iris(return_X_y=True)

    lr = LogisticRegression()

    print(lr.get_search_space())

    grid = GridSearchCV(lr, param_grid=lr.get_search_space())
    grid.fit(x, y)

    print(grid.score(x, y))
    print(grid.best_estimator_)

    print(hasattr(lr, 'get_search_space'))

    print(lr.name)
