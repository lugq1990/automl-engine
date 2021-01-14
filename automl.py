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
from auto_ml.utils.data_rela import check_data_and_label, hash_dataset_name, check_label
from auto_ml.base.classifier_algorithms import *
from auto_ml.metrics.scorer import *
from auto_ml.utils.CONSTANT import *
from auto_ml.pipelines.pipeline_training import ClassificationPipeline
from auto_ml.utils.func_utils import deprecated
from auto_ml.utils.logger import logger
from auto_ml.ensemble.model_ensemble import ModelEnsemble


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

        # Deprecated: We should ensure the whole thing should happen in pipeline `class`
        # for whole algorithms should be loaded first
        # self.algorithm_dir = load_yaml_file("default_algorithms.yml")
        # this is to save the whole instance object, so here should be in parent logic.
        # So child just to implement with function: `_create_ml_object_dir`
        # self.al_obj_dir = dict()
        # NOTED: this should be replaced with pipeline training step.
        # self._create_ml_object_dir()

        self.estimator = None

        # This is used to do testing and prediction.
        self.models_list = self._load_trained_models_ordered_by_score()
        # as we will use `ensemble` to combine models so the last will just be one model
        self.best_model = None

        # Add with what type of the problem
        self.type_of_problem = None

    def fit(self, xtrain,
            ytrain,
            metric: Scorer,
            task=None,
            xtest=None,
            ytest=None,
            batch_size=None,
            dataset_name=None):
        """
        Type of the problem attribute should be added, so that for `score`,
        we could get metrics based on different problem.
        :param xtrain:
        :param ytrain:
        :param metric:
        :param task:
        :param xtest:
        :param ytest:
        :param batch_size:
        :param dataset_name:
        :return:
        """
        raise NotImplementedError

    def _check_fitted(self):
        if not self.models_list:
            logger.error("When try to get prediction with `automl`, Please"
                                 "fit the model first")
            raise NotFittedError("When try to get prediction with `automl`, Please"
                                 "fit the model first")

        return True

    def predict(self, x, **kwargs):
        """
        Most of the prediction logic should happen here, as for the score should based on
        prediction result.

        :param x:
        :param kwargs:
        :return:
        """
        logger.info("Start to get prediction based on best trained model!")

        # first should to process data
        x_processed = self._process_data_with_trained_processor(x)

        logger.info("Start to get prediction based on best trained model.")
        prediction = self.best_model.predict(x_processed)
        logger.info("Prediction step finishes.")

        return prediction

    def predict_proba(self, x, **kwargs):
        """
        Should support with probability if supported.
        :param x:
        :param kwargs:
        :return:
        """
        logger.info("Start to get probability based on best trained model!")

        # should check the estimator should have function: `predict_proba`!
        if not hasattr(self.best_model, 'predict_proba'):
            raise NotImplementedError("Best fitted model:{} doesn't support `predict_proba`".format(self.best_model))

        logger.info("Start to process data with trained processor pipeline.")
        x_processed = self._process_data_with_trained_processor(x)

        logger.info("Start to get probability based on best trained model.")
        prob = self.best_model.predict_proba(x_processed)
        logger.info("Prediction step finishes.")

        return prob

    def score(self, x, y, **kwargs):
        self._check_fitted()

        logger.info("Start to get prediction based on best trained model!")

        # first should get prediction, then we could get different score based on the prediction
        prediction = self.predict(x)

        # for different problem use different scorer
        scorer = self._get_scorer_based_on_target(y)

        logger.info("Start to use metrics: {} to get score.".format(scorer))
        score = scorer(y, prediction)

        logger.info("Get score: {} using metrics: {}".format(score, scorer))

        return score

    def _process_data_with_trained_processor(self, data):
        """
        Let this function to do whole processing data logic.
        :param data: 
        :return: 
        """
        # This func is used only after the model is fitted.
        self._check_fitted()
        
        logger.info('Start to load trained processor from disk.')
        processor = self.backend.load_model('processing_pipeline')

        try:
            # here I think to get prediction should based on highest score model
            # for classifiction should `acc` highest as 1th, for regression should `rmse` lowest
            # as 1th, but any way should use first model
            self.best_model = self.models_list[0][1]
            
            logger.info("Start to process data with `trained processor pipeline`.")
            data_new = processor.transform(data)

            # the `stacking` logic should happen here, for upper we could just use it, don't need to care detail
            if "stacking" in self.models_list[0][0].lower():
                logger.info("Creating new dataset based on trained models with stacking.")
                # process data based on the `stacking` model logic.
                data_new = ModelEnsemble.create_stacking_dataset(data_new)

            return data_new
        except Exception as e:
            raise Exception("When try to process data with "
                            "trained processor pipeline get error: {}".format(e))

    def _get_scorer_based_on_target(self, y):
        # for different problem use different scorer
        self.type_of_problem = self._get_type_problem(y)
        logger.info("Get type of problem: {}".format(str(self.type_of_problem)))

        if self.type_of_problem == 'classification':
            scorer = accuracy
        elif self.type_of_problem == 'regression':
            scorer = mean_squared_error
        else:
            raise ValueError("When to score data get not "
                             "supported type of problem: {}".format(self.type_of_problme))

        return scorer

    @staticmethod
    def _get_type_problem(y):
        """
        To check what type of the label dataset.
        :param y:
        :return:
        """
        label_type = check_label(y)

        if label_type in CLASSIFICTION_TASK:
            return 'classification'
        elif label_type in REGRESSION_TASK:
            return 'regression'
        else:
            raise ValueError("When to check label type based label data, "
                             "get not supported type: {}".format(label_type))

    @deprecated
    def _create_ml_object_dir(self):
        """create a list of object that we need,
        here is to use the list of names that we need to instant
        each algorithm class, so that we could start the training
        step using these algorithms.

        Let subclass to implement.
        the directory should be {name: instance_obj}.
        I think to put it into the init func will be better."""
        raise NotImplementedError

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


class ClassificationAutoML(AutoML):
    def __init__(self):
        super(ClassificationAutoML, self).__init__()
        # after pipeline has finished, then we should use `ensemble` to combine these models
        # action should happen here.
        self.estimator = ClassificationPipeline()

    def fit(self, xtrain, ytrain, task=None, metric=accuracy,
            xtest=None, ytest=None, batch_size=None, dataset_name=None):
        """
        I think that we could use sub-class to do real training.
        Also I think that for parent could just give the direction,
        should be implemented here.

        :param xtrain: train data
        :param ytrain: label data
        :param task: should be in [0, 1, 2]
        :param metric: how to evaluate result
        :param xtest:test data
        :param ytest: test label
        :param batch_size: how much data for training
        :return:
        """
        # first we should get the task type, that's for what metrics to use.
        classification_tasks = [0, 1, 2]
        if task is None:
            # we should get the task type by self.
            task = self._get_task_type(ytrain)
        elif task not in classification_tasks:
            raise ValueError("Task should be in [{}]".format(' '.join(classification_tasks)))

        # Type of the problem should happen here.
        self.type_of_problem = 'classifiction'

        self.dataset_name = hash_dataset_name(xtrain) if dataset_name is None else dataset_name
        # we should ensure data could be trained like training data should be 2D
        xtrain, ytrain = check_data_and_label(xtrain, ytrain)

        # after the checking process, then we need to create the Pipeline for whole process.
        # Here should use a Pipeline object to do real training, also with `ensemble`
        self.estimator.fit(xtrain, ytrain)

        # load trained models for prediction and scoring for testing data.
        # after we have fitted the trained models, then next step is to load whole of them from disk
        # and sort them based on score, and get scores for `test data` and `test label`
        # WE could get them from parent function, so that we could also use this for `regression`
        self.models_list = self._load_trained_models_ordered_by_score(higher_best=True)

    @staticmethod
    def _get_task_type(y):
        """
        This is to get the target type that we want to do.
        :param y: label data
        :return:
        """
        if len(y.shape) == 2:
            # label with 2D that's multi-label
            # raise ValueError("Target should be just 1D for sklearn")
            return 2
        else:
            unique_values = np.unique(y)
            if len(unique_values) == 2:
                return 0
            else:
                return 1


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)

    auto_cl = ClassificationAutoML()
    # auto_cl.fit(x, y)

    print(auto_cl.models_list)
    print(auto_cl.score(x, y))
    print('*' * 20)
    print(auto_cl.predict(x))
    print('*'*20)
    print(auto_cl.predict_proba(x))
