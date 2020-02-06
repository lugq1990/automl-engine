# -*- coding:utf-8 -*-
"""
This class is the main class that whole sklearn
estimators should inherit from this class, and
this class will set with the training step information,
like which folder to save the trained model, how much
time that total model training step, each algorithm
could be fitted time and how big the data set or something
useful in this class.
author: Guangqiang.lu
"""
from sklearn.base import BaseEstimator
from ..backend_sklearn.automl import ClassificationAutoML


class Estimators(BaseEstimator):
    """
    based on sklearn base estimator that we could fix
    whole estimators.
    """
    def __init__(self, total_times=3600,
                 each_task_time=300,
                 ensemble_size=5,
                 ensemble_nbest_size=10,
                 data_memory_limit=1024,
                 include_estimators=None,
                 include_processors=None,
                 exclude_estimators=None,
                 exclude_processors=None,
                 models_folder=None,
                 tmp_models_folder=None,
                 delete_tmp_models_folder_after_finish=True,
                 delete_models_folder_after_finish=False,
                 n_jobs=None,
                 ):
        """
        This is main class for whole sklearn processor class.

        :param total_times: int type, default=3600
            how much time that could be used to train models.
        :param each_task_time: int type, default=300
            how much time for each algorithm that could be trained.
        :param ensemble_size: int type, default=5
            how many ensembles models could be used to make the ensemble
            models, if make this bigger, the more chance to find a better
            model.
        :param ensemble_nbest_size: int type, default=10
            how many already trained model could be used to make the final
            ensemble model.
        :param data_memory_limit: int type, default=1024
            how big data could be used for training models.
        :param include_estimators: list(object), default=None
            which algorithms could just be used to fit the model, if default,
            then whole models will be fitted, if with these algorithms chosen,
            then just use these algorithms.
        :param include_processors: list(object), default=None
            which processing algorithms should be used to process the data.
            default will use whole predefined algorithms to process the data,
            otherwise will just use the chosen algorithms.
        :param exclude_estimators: list(object), default=None
            which algorithms should be excluded from the training algorithms,
            if these algorithms are chosen, then just remove these algorithms
            that could be used to fit the model.
        :param exclude_processors: list(object), default=None
            which processing algorithms could be excluded for the data, in case
            that we don't want to use to process the data.
        :param models_folder: str type, default=None
            the local folder the should be used to store the final combined model
            that could be used to put it into production.
        :param tmp_models_folder: str type, default=None
            which temperate folder could be used for whole training step to store
            the already trained model.
        :param delete_tmp_models_folder_after_finish: Boolean, default=True
            whether or not to delete the temperate folder for the models folder,
            default is true to delete the whole folders for stepping models.
        :param delete_models_folder_after_finish: Boolean, default=False
            whether or not to delete the already trained final ensemble or selected
            models that are stored in that folder, default is not to delete the folder.
        :param n_jobs: int type, default=None
            how many processors should be used for training step, default is None means
            to use whole processors, if negative then will also use whole processors.
        """
        self.total_times = total_times
        self.each_task_time = each_task_time
        self.ensemble_size = ensemble_size
        self.ensemble_nbest_size = ensemble_nbest_size
        self.data_memory_limit = data_memory_limit
        self.include_estimators = include_estimators
        self.include_processors = include_processors
        self.exclude_estimators = exclude_estimators
        self.exclude_processors = exclude_processors
        self.models_folder = models_folder
        self.tmp_models_folder = tmp_models_folder
        self.delete_tmp_models_folder_after_finish = delete_tmp_models_folder_after_finish
        self.delete_models_folder_after_finish = delete_models_folder_after_finish
        if n_jobs < 0:
            n_jobs = None
        self.n_jobs = n_jobs
        self.estimator = None
        super().__init__()

    def fit(self, **kwargs):
        """
        This the the whole step that the model could be trained using this function.
        args could be specified with sub-class.
        :param kwargs: key-words could be used to fit the model.
        :return: self
        """
        pass

    def predict(self, **kwargs):
        pass

    def score(self, **kwargs):
        pass

    @staticmethod
    def get_estimator():
        pass


class ClassificationEstimator(Estimators):
    def fit(self, xtrain,
            ytrain,
            xtest=None,
            ytest=None,
            n_jobs=None,
            batch_size=None):
        if n_jobs is None or n_jobs == 1:
            # with just one thread to do training
            # we have to instant the object.
            self.estimator = self.get_estimator()()
            # ensure data should be array type
            self.estimator.fit(xtrain, ytrain, xtest, ytest, batch_size)

        else:
            # we get n_jobs, we should use parallel training step.
            pass

    def predict(self, xtest, ytest, batch_size=None):
        pass

    def score(self, xtest, ytest, batch_size=None):
        pass

    @staticmethod
    def get_estimator():
        """
        this is to get the classification auto-ml object, so that
        we could use this object to do training.
        :return: class pointer
        """
        return ClassificationAutoML
