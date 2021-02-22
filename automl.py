# -*- coding:utf-8 -*-
"""
This is a class that is used for creating auto-ml object that
we could use for training step.

author: Guangqiang.lu
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from auto_ml.utils.backend_obj import Backend
from auto_ml.base.classifier_algorithms import *
from auto_ml.metrics.scorer import *
from auto_ml.utils.CONSTANT import *
from auto_ml.pipelines.pipeline_training import ClassificationPipeline
from auto_ml.utils.logger import create_logger


logger = create_logger(__file__)


class AutoML(BaseEstimator):
    def __init__(self, backend=None,
                 time_left_for_this_task=3600,
                 n_ensemble=10,
                 n_best_model=5,
                 include_estimators=None,
                 exclude_estimators=None,
                 include_preprocessors=None,
                 exclude_preprocessors=None,
                 keep_models=True,
                 model_dir=None,
                 precision=32,
                 ):
        """
        this is to init automl class, whole thing should be ininstanted
        in this class, like what algorithms to use, how many models to
        be selected, etc.
        :param backend: backend object used to save and load models
        :param time_left_for_this_task: how long for this models to be trained.
        :param n_ensemble: how many models to be selected to be ensemble
        :param n_best_model: how many models to be keeped during training.
        :param include_estimators: what algorithms to be included
        :param exclude_estimators: what algorithms to be excluded
        :param include_preprocessors: what preprocessing step to be included
        :param exclude_preprocessors: what preprocessing step to be excluded
        :param keep_models: whether or not to keep trained models
        :param model_dir: keep model folder, if None use backend to create one folder
        :param precision: precision of data, to save memory
        """
        super(AutoML, self).__init__()
        self.backend = Backend() if backend is None else backend
        self.time_left_for_this_taks = 3600 if time_left_for_this_task is None else time_left_for_this_task
        self.n_ensemble = n_ensemble
        self.n_best_model = n_best_model
        self.include_estimators = include_estimators
        self.exclude_estimators = exclude_estimators
        self.include_preprocessors = include_preprocessors
        self.exclude_preprocessors = exclude_preprocessors
        self.keep_models = keep_models
        self.model_dir = model_dir
        self.precision = precision
        self.estimator = None

        # This is used to do testing and prediction.
        self.models_list = self._load_trained_models_ordered_by_score()
        # as we will use `ensemble` to combine models so the last will just be one model
        self.best_model = None

        # Add with what type of the problem
        self.type_of_problem = None

    def fit(self, xtrain,
            ytrain,
            n_jobs=None):
        """
        Type of the problem attribute should be added, so that for `score`,
        we could get metrics based on different problem.
        :param xtrain:
        :param ytrain:
        :param n_jobs:
        :return:
        """
        raise NotImplementedError

    def predict(self, x, **kwargs):
        """
        Most of the prediction logic should happen here, as for the score should based on
        prediction result.

        :param x:
        :param kwargs:
        :return:
        """
        logger.info("Start to get prediction based on best trained model.")
        prediction = self.estimator.predict(x)
        logger.info("Prediction step finishes.")

        return prediction

    def predict_proba(self, x, **kwargs):
        """
        Should support with probability if supported.
        :param x:
        :param kwargs:
        :return:
        """
        # should check the estimator should have function: `predict_proba`!
        if not hasattr(self.estimator, 'predict_proba'):
            raise NotImplementedError("Best fitted model:{} doesn't support `predict_proba`".format(self.best_model))

        logger.info("Start to get probability based on best trained model.")
        prob = self.estimator.predict_proba(x)
        logger.info("Prediction step finishes.")

        return prob

    def score(self, x, y, **kwargs):
        self._check_fitted()

        logger.info("Start to get prediction based on best trained model!")

        # Use child func.
        score = self.estimator.score(x, y)

        logger.info("Get score: {} based on best model!".format(score))

        return score

    def _load_trained_models_ordered_by_score(self, higher_best=True):
        """
        To load whole trained model from disk and sorted them based on `higher_best`.
        :param higher_best:
        :return:
            models_list: a list of trained models sorted with score.
        """
        models_list = self.backend.load_models_combined_with_model_name()

        # let's sort these models based on score, this is a tuple:(model_name, model_instance)
        if higher_best:
            reverse = True
        else:
            reverse = False

        models_list = sorted(models_list,
                             key=lambda model: float(model[0].split("-")[1].replace(".pkl", '')),
                             reverse=reverse)

        return models_list

    def get_sorted_models_scores(self, x, y, **kwargs):
        """
        Add this func to get whole score based for each trained models, so that
        we could get the result that we have taken that times and for each models,
        how about the testing result.

        But this should be implemented by pipeline!
        :param x:
        :param y:
        :param kwargs:
        :return:
            a list of tuple: [(model_name, model_score), ...]
        """
        raise NotImplementedError

    def _check_fitted(self):
        if not self.models_list:
            logger.error("When try to get prediction with `automl`, Please fit the model first")
            raise NotFittedError("When try to get prediction with `automl`, Please fit the model first")

        return True


class ClassificationAutoML(AutoML):
    def __init__(self):
        super(ClassificationAutoML, self).__init__()
        # after pipeline has finished, then we should use `ensemble` to combine these models
        # action should happen here.
        self.estimator = ClassificationPipeline()

    def fit(self, xtrain, ytrain, n_jobs=None, use_neural_network=True):
        """
        I think that we could use sub-class to do real training.
        Also I think that for parent could just give the direction,
        should be implemented here.

        :param xtrain: train data
        :param ytrain: label data
        :param n_jobs: how many cores to use
        :return:
        """
        # after the checking process, then we need to create the Pipeline for whole process.
        # Here should use a Pipeline object to do real training, also with `ensemble`
        self.estimator.fit(xtrain, ytrain, n_jobs=n_jobs, use_neural_network=use_neural_network)

        # load trained models for prediction and scoring for testing data.
        # after we have fitted the trained models, then next step is to load whole of them from disk
        # and sort them based on score, and get scores for `test data` and `test label`
        # WE could get them from parent function, so that we could also use this for `regression`
        self.models_list = self._load_trained_models_ordered_by_score(higher_best=True)

    def get_sorted_models_scores(self, x, y, **kwargs):
        """
        To get some best trained model's score for `test` data with ordered.

        So that we could get the list of the best scores for later front end show case.
        :param x:
        :param y:
        :param kwargs:
        :return:
        """
        score_dict = self.estimator.get_sorted_models_scores(x, y, reverse=True)

        return score_dict


if __name__ == '__main__':
    from auto_ml.test.get_test_data import get_training_data

    df = get_training_data(return_df=True)

    x = df.drop(['Survived'], axis=1).values
    y = df['Survived'].values
    # needed to be added for the array when we read data from a file.
    x = x.copy(order='C')

    from sklearn.model_selection import train_test_split

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)

    auto_cl = ClassificationAutoML()
    auto_cl.fit(xtrain, ytrain)

    print(auto_cl.models_list)
    print(auto_cl.score(xtest, ytest))
    print('*' * 20)
    print(auto_cl.predict(xtest)[:10])
    print('*'*20)
    print(auto_cl.predict_proba(xtest)[:10])

    # get model score
    print(auto_cl.get_sorted_models_scores(xtest, ytest))
