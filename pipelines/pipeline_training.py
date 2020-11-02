# -*- coding:utf-8 -*-
"""
This will just contain the training logic here with both preprocessing and algorithm training.

@author: Guangqiang.lu
"""
from sklearn.pipeline import make_pipeline, Pipeline
from auto_ml.preprocessing import *
from auto_ml.base import model_selection


class PipelineTrain(Pipeline):
    """
    Let's make it as parent class for both classification and regression.
    """
    def __init__(self,
                 use_standard=True,
                 use_norm=True,
                 use_pca=False,
                 use_minmax=False,
                 user_feature_seletion=True
                 ):
        self.use_standard = use_standard
        self.use_norm = use_norm
        self.use_pca = use_pca
        self.use_minmax = use_minmax
        self.user_feature_seletion = user_feature_seletion
        self.pipeline = None

    def build_pipeline(self):
        """
        Real pipeline step should happen here.
        Let child to do real build.
        :return:
        """
        raise NotImplementedError

    def fit(self, X, y=None, **fit_params):
        try:
            self.pipeline.fit(X, y)
            return self
        except Exception as e:
            raise Exception("When do real pipeline training get error: {}".format(e))

    def score(self, x, y):
        return self.pipeline.score(x, y)

    def predict(self, x):
        return self.pipeline.predict(x)

    def predict_proba(self, x):
        return self.pipeline.predict_proba(x)



