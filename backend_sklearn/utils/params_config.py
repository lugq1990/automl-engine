# -*- coding:utf-8 -*-
"""
this is to write a class that we could use for
whole algorithms that we used in this class.

@author: Guangqiang.lu
"""
import numpy as np
import os

from backend.backend_sklearn.utils.paths import load_param_config


class ConfigSpace(object):
    def __init__(self):
        self.config = load_param_config()
        self.algorithm_config = None

    def load_sklearn_params(self, algorithm_name):
        if algorithm_name == 'LogisticRegression':
            self.algorithm_config = self._load_lr_params()
        else:
            pass

    def _load_lr_params(self):
        config_lr = self.config['sklearn']['classification']['LogisticRegression']
        config_lr['C'] = np.random.exponential(1, size=5)
        config_lr['fit_intercept'] = [True, False]
        config_lr['penalty'] = ['l1', 'l2']

        return config_lr


if __name__ == '__main__':
    config = ConfigSpace()
    config.load_sklearn_params('LogisticRegression')
    lr_config = config.algorithm_config

    from backend.backend_sklearn.base.classfication.logistic_regression import LR
    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)

    lr = LR()
    estimator = GridSearchCV(lr, lr_config, cv=10)
    estimator.fit(x, y)
    print(estimator.best_score_)
