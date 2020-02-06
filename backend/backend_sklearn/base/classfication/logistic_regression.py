# -*- coding:utf-8 -*-
"""
This is base estimator that we could use to create
a Logistic Regression model, in fact we do some
hyper-parameters setting for this model

@author: Guangqiang.lu
"""
from sklearn.linear_model import LogisticRegression


class LR():
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
        self.estimator = LogisticRegression(C=self.C,
                                class_weight=self.class_weight,
                                dual=self.dual,
                                fit_intercept=self.fit_intercept,
                                penalty=self.penalty,
                                n_jobs=self.n_jobs,
                                random_state=self.random_state)

        self.estimator.fit(x, y)

    def predict(self, x):
        return self.estimator.predict(x)

    def score(self, x, y):
        return self.estimator.score(x, y)

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)


if __name__ == '__main__':
    lr = LR()

    from sklearn.datasets import load_iris
    x, y  = load_iris(return_X_y=True)

    lr.fit(x, y)

    print(lr.score(x, y))

