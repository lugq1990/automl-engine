# -*- coding:utf-8 -*-
"""
To do ensemble logic with whole trained model, try to improve whole score based on
different processing logic, also if we could get better result, then we are lucky!

One more thing, this should be called only and after the pipeline has finished, so that
we could load the trained model from disk, so this should be called from the parent
automl training logic.

@author: Guangqiang.lu
"""
import numpy as np
from sklearn.base import BaseEstimator
from auto_ml.utils.backend_obj import Backend
from auto_ml.metrics.scorer import accuracy, r2


class ModelEnsemble(BaseEstimator):
    """
    Currently I want to support 2 different ensemble logic:
    bagging(weight combine classification: voting, regression: weight multiple)
    stacking(add trained model prediction into training data)
    """

    def __init__(self, task_type='classification', ensemble_alg='bagging'):
        """
        Based on different task to do different logic.
        :param task_type: which task to do: classification or regression.
        :param ensemble_alg:
        """
        self.task_type = task_type
        self.ensemble_alg = ensemble_alg
        # To load and save models
        self.backend = Backend()
        self.model_list = self._load_trained_models()
        # define matrics based on task
        self.metrics = None
        if self.task_type == 'classification':
            self.metrics = accuracy
        elif self.task_type == 'regression':
            self.metrics = r2
        # self.dataset = self.backend.load_dataset('processed_data')

    def fit(self, x, y, xtest, ytest, **kwargs):
        if self.ensemble_alg == 'bagging':
            self.fit_bagging(x, y, xtest, ytest, **kwargs)
        elif self.ensemble_alg == 'stacking':
            self.fit_stacking(x, y, xtest, ytest, **kwargs)

    def fit_bagging(self, x, y, xtest, ytest, **kwargs):
        """
        For bagging fitting, if we face with classification problem,
        then we could use `voting` logic to get ensemble prediction,
        if regression, then will get weights * each model prediction.
        """
        if self.task_type == 'classification':
            # based on prediction and do voting logic
            pred_list = []
            for model in self.model_list:
                # get each model prediction, convert into list
                pred = model.predict(xtest).tolist()
                pred_list.append(pred)
            # convert a list of prediction into array for later
            pred_arr = np.array(pred_list)

            # get most frequent values based on voting logic to get prediction
            xtest_pred = np.array([np.bincount(pred_arr[:, i]).argmax() for i in range(pred_arr.shape[1])])

            score = self.metrics(ytest, xtest_pred)


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
        model_list.sort(key=lambda x: float("0." + x[0].split('-')[1].split('.')[0]), reverse=True)
        # model_list = sorted(model_list, key=lambda x: x[0].split('.')[0].split('-')[1])
        return model_list

    def get_model_score_list(self):
        """
        To get each model accuracy score list for later compare
        :return:
        """
        score_list = []

        for model_name, _ in self._load_trained_models():
            model_score = model_name.split('.')[0].split('-')[-1]
            score_list.append(model_score)

        return score_list


if __name__ == '__main__':
    model_ensemble = ModelEnsemble()
    print(model_ensemble.model_list)


