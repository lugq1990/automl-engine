# -*- coding:utf-8 -*-
"""
Here is to select some features if we have so many features, one thing that we could
reduce our training time, also we could remove some relevant features.

@author: Guangqiang.lu
"""
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from auto_ml.preprocessing.processing_base import Process
from auto_ml.utils.data_rela import check_label
from auto_ml.utils.CONSTANT import *


class FeatureSelect(Process):
    def __init__(self, simple_select=True, tree_select=False):
        super(FeatureSelect, self).__init__()
        self.simple_select = simple_select
        self.tree_select = tree_select

    def fit(self, data, label=None):
        """
        Also support with algorithm based feature selection
        :param data: data to process
        :param label: label data if need with algorithm trained based
        :return:
        """
        if self.simple_select:
            self.estimator = VarianceThreshold()
            self.estimator.fit(data)
        else:
            # if we want to use algorithms based feature extraction, currently will use LinearRegression
            if label is None:
                raise ValueError("When want to use Algorithm, label data should be provided!")

            # before next step, we should ensure task should be classification
            label_type = check_label(label)
            if STRING_TO_TASK.get(label_type) not in CLASSIFICTION_TASK:
                raise ValueError("When we want to use model selection logic, task type should just be classification.")

            if self.tree_select:
                model = ExtraTreesClassifier(n_estimators=50).fit(data, label)
            else:
                model = LinearRegression().fit(data, label)

            self.estimator = SelectFromModel(model, prefit=True)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x,y = load_iris(return_X_y=True)

    f = FeatureSelect(simple_select=False)
    f.fit(x, y)
    print(f.transform(x))
