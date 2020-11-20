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
from sklearn.ensemble import VotingClassifier, VotingRegressor
from auto_ml.utils.backend_obj import Backend
from auto_ml.metrics.scorer import accuracy, r2


class ModelEnsemble(BaseEstimator):
    """
    Currently I want to support 2 different ensemble logic:
    Voting(weight combine classification: with soft voting and hard voting,
    regression: weight multiple)
    stacking(add trained model prediction into training data)
    """
    def __init__(self, task_type='classification', ensemble_alg='voting',
                 voting_logic='soft'):
        """
        Based on different task to do different logic.
        :param task_type: which task to do: classification or regression.
        :param ensemble_alg: which ensemble logic to use
        :param voting_logic: whether with `hard` or `soft` voting
        """
        self.task_type = task_type
        self.ensemble_alg = ensemble_alg
        self.voting_logic = voting_logic
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
        if self.ensemble_alg == 'voting':
            self.fit_bagging(x, y, xtest, ytest, **kwargs)
        elif self.ensemble_alg == 'stacking':
            self.fit_stacking(x, y, xtest, ytest, **kwargs)

    def fit_bagging(self, x, y, xtest, ytest, **kwargs):
        """
        Here with ensemble logic like `hard` by number voting or
        `soft` by weight combine.

        For bagging fitting, if we face with classification problem,
        then we could use `voting` logic to get ensemble prediction,
        if regression, then will get weights * each model prediction.
        """

        # Here change logic with voting
        if self.task_type == 'classification':
            model_list_without_score = self._get_model_list_without_score()
            if self.voting_logic not in ['hard', 'soft']:
                raise ValueError("For ensemble logic, only `hard` and soft is supported "
                                 "when use `voting` logic.")

            voting_estimator = VotingClassifier(estimators=model_list_without_score,
                                                voting=self.voting_logic)

            # start to fit the voting estimator
            voting_estimator.fit(x, y)

            # get voting model score
            score = voting_estimator.score(xtest, ytest)
            score_str = str(round(score, 6)).split('.')[-1]

            store_model_name = 'Voting_{}-{}'.format(self.voting_logic, score_str)

            print('voting score:', score)
            self.backend.save_model(voting_estimator, store_model_name)

        elif self.task_type == 'regression':
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

        # ADD logic: we shouldn't include the `Voting` algorithms instance object in fact
        model_list = [x for x in model_list if not x[0].lower().startswith('voting')]

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

    def _get_model_list_without_score(self):
        """
        To get the model list without score for ensemble use case.
        :return:
        """
        model_list_without_score = []
        for estimator_tuple in self.model_list:
            estimator_name = estimator_tuple[0].split('-')[0]
            model_list_without_score.append((estimator_name, estimator_tuple[1]))

        return model_list_without_score


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)

    model_ensemble = ModelEnsemble(voting_logic='soft')

    # model_ensemble.fit(x, y, x, y)
    # print([x[1].__class__ for x in model_ensemble.model_list])
    print(model_ensemble.model_list)
    # for models in model_ensemble.model_list:
    #     model = models[1]
    #     print(model)
    #     print(getattr(model, "_estimator_type", None))


    model_ensemble.fit(x, y, x, y)
