# -*- coding:utf-8 -*-
"""
To do ensemble logic with whole trained model, try to improve whole score based on
different processing logic, also if we could get better result, then we are lucky!

One more thing, this should be called only and after the pipeline has finished, so that
we could load the trained model from disk, so this should be called from the parent
automl training logic.

@author: Guangqiang.lu
"""
from sklearn.base import BaseEstimator
from auto_ml.utils.backend_obj import Backend


class ModelEnsemble(BaseEstimator):
    """
    Currently I want to support 2 different ensemble logic:
    bagging(weight combine classification: voting, regression: weight multiple)
    stacking(add trained model prediction into training data)
    """

    def __init__(self, ensemble_alg='bagging'):
        self.ensemble_alg = ensemble_alg
        # To load and save models
        self.backend = Backend()
        self.model_list = self._load_trained_models()

    def fit(self, x, y, **kwargs):
        if self.ensemble_alg == 'bagging':
            self.fit_bagging(x, y, **kwargs)
        elif self.ensemble_alg == 'stacking':
            self.fit_stacking(x, y, **kwargs)

    def fit_bagging(self, x, y, **kwargs):
        pass

    def fit_stacking(self, x, y, **kwargs):
        pass

    def _load_trained_models(self):
        """
        To load whole trained model from disk, one more thing, as we also saved
        the processing model into disk, so here should ensure we just load the
        algorithm models.

        Sorted instance object list with name, also we could get the model score
        for later compare use case.
        :return:
        """
        model_list = self.backend.load_models_combined_with_model_name()

        # after we have get the model list, we should ensure the model by the model
        # name with endswith score.
        # Model name like this: ('lr_98.pkl', lr)
        model_list.sort(key=lambda x: x[0][x.index('-') + 1: x.index('.')], reverse=True)

        return model_list




