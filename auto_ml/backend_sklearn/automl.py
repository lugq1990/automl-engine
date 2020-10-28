# -*- coding:utf-8 -*-
"""
This is a class that is used for creating auto-ml object that
we could use for training step.

author: Guangqiang.lu
"""
import numpy as np
from sklearn.base import BaseEstimator
from auto_ml.backend.backend_sklearn.utils.backend_obj import Backend
from auto_ml.backend.backend_sklearn.metrics.scorer import Scorer
from auto_ml.backend.backend_sklearn.utils.data_rela import check_data_and_label, hash_dataset_name
from auto_ml.backend.backend_sklearn.utils.files import load_yaml_file


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

    def fit(self, xtrain,
            ytrain,
            task: int,
            metric: Scorer,
            xtest: np.array = None,
            ytest: np.array = None,
            dataset_name = None):
        # first check data and label to ensure data and
        # label as we want.
        if not isinstance(task, int):
            raise ValueError("We have to use int type of task!")

        # we want to store the dataset name as a string.
        self.dataset_name = hash_dataset_name(xtrain) if dataset_name is None else dataset_name

        xtrain, ytrain = check_data_and_label(xtrain, ytrain)

    def predict(self, x, **kwargs):
        pass

    def score(self, x, y, **kwargs):
        pass

    def create_ml_object_dir(self):
        """create a list of object that we need,
        here is to use the list of names that we need to instant
        each algorithm class, so that we could start the training 
        step using these algorithms.
        Let subclass to implement.
        the directory should be {name: instance_obj}"""
        raise NotImplementedError

    def _get_training_algorithm_names(self, file_name):
        """according to different problem to 
        get default algorithms"""

        self.algorithm_dir = load_yaml_file(file_name)


class ClassificationAutoML(AutoML):
    def __init__(self):
        super(ClassificationAutoML, self).__init__()

    def fit(self, xtrain, ytrain, xtest=None, ytest=None, batch_size=None):
        pass

    def create_ml_object_dir(self):
        algorithm_name_list = self.algorithm_dir['classification']['default']
        from auto_ml.backend.backend_sklearn.base.classifier_algorithms import *

        al_obj_dir = {}
        # currently with manually creation of object
        for al in algorithm_name_list:
            if al == 'LogisticRegression':
                al_obj_dir[al] = LogisticRegression()
            elif al == 'SupportVectorMachine':
                al_obj_dir[al] = SupportVectorMachine()

