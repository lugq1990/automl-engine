# -*- coding:utf-8 -*-
"""
This is used for whole classification problem that could be used
with sklearn backend engine.
author: Guangqiang.lu
"""
from auto_ml.estimator import ClassificationEstimator


if __name__ == '__main__':
    # this is unittest for classification functionality.
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    x, y = load_iris(return_X_y=True)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, random_state=123)

    estimator = ClassificationEstimator()

    estimator.fit(xtrain, ytrain, xtest, ytest)

    print(estimator.score(xtest, ytest))


