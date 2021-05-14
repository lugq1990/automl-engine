# -*- coding:utf-8 -*-
"""
This is a class that is used for creating auto-ml object that
we could use for training step.

author: Guangqiang.lu
"""

import numpy as np
import pandas as pd
import time
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from auto_ml.utils.backend_obj import Backend
from auto_ml.base.classifier_algorithms import *
from auto_ml.metrics.scorer import *
from auto_ml.utils.CONSTANT import *
from auto_ml.pipelines.pipeline_training import ClassificationPipeline
from auto_ml.utils.logger import create_logger


logger = create_logger(__file__)


class AutoML(BaseEstimator):
    def __init__(self, models_path=None,
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
                 delete_models=True
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
        # Change backend object creation as this class is entry class also Backend class support singleton, so output_folder will work.
        models_path = OUTPUT_FOLDER if models_path is None else models_path
        # Add with `models_path` parameters, as the only output for the framework is only models files, so to store models into
        # folder that we would like will be nice!
        self.models_path = models_path
        
        self.backend = Backend(output_folder=models_path) 
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

        # as we will use `ensemble` to combine models so the last will just be one model
        self.best_model = None

        # Add with what type of the problem
        self.type_of_problem = None

        self.delete_models = delete_models

    def fit(self, *args, **kwargs):
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
            models_list: a list of trained models sorted with score. [("lr-0.9.pkl", lr-0.9.pkl), ...]
        """
        models_list = self.backend.load_models_combined_with_model_name()

        # let's sort these models based on score, this is a tuple:(model_name, model_instance)
        if higher_best:
            reverse = True
        else:
            reverse = False

        # sort models by `training` score with diff type of problem
        models_list = sorted(models_list,
                             key=lambda model: float(model[0].split("-")[1].replace(".pkl", '').replace(".h5", '')),
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
    def __init__(self, models_path=None, 
            include_estimators=None, 
            exclude_estimators=None, 
            include_preprocessors=None, 
            exclude_preprocessors=None,
            **kwargs):
        """Added with algorithm selection and processing selection, even with others in case we need.

        Args:
            models_path (Str, optional): Where to store our models. Defaults to None.
        """
        super(ClassificationAutoML, self).__init__(models_path=models_path,
                                            include_estimators=include_estimators,
                                            exclude_estimators=exclude_estimators,
                                            include_preprocessors=include_preprocessors,
                                            exclude_preprocessors=exclude_preprocessors, **kwargs)
        # after pipeline has finished, then we should use `ensemble` to combine these models
        # action should happen here.
        self.estimator = ClassificationPipeline(backend=self.backend,
                                            include_estimators=include_estimators,
                                            exclude_estimators=exclude_estimators,
                                            include_preprocessors=include_preprocessors,
                                            exclude_preprocessors=exclude_preprocessors, 
                                            **kwargs)

    def fit(self, file_load=None, x=None, y=None, \
             xval=None, yval=None, val_split=None, n_jobs=None, use_neural_network=True):
        """
        Real training logic happen here, also store trained models.

        Support both with file to train also with data and label data.

        :param xtrain: train data
        :param ytrain: label data
        :param n_jobs: how many cores to use
        :return:
        """
        start_time = time.time()

        # Add logic here is that we should clean models' folder, so that every time we could get a clean folder for next time running!
        if self.backend and self.delete_models:
            self.backend.clean_folder()

        x, y = self._get_data_and_label(file_load, x, y)

        if val_split is not None:
            # if do need to do validation based on current train data, then just split current data into validation as well
            x, xval, y, yval = train_test_split(x, y, test_size=val_split)
        else:
            # Here I think if and only if the data length is over a threashold, then to do validation,even user haven't provided val data.
            if len(x) > VALIDATION_THRESHOLD:
                val_split = .2
                # if do need to do validation based on current train data, then just split current data into validation as well
                x, xval, y, yval = train_test_split(x, y, test_size=val_split)

        # after the checking process, then we need to create the Pipeline for whole process.
        # Here should use a Pipeline object to do real training, also with `ensemble`
        self.estimator.fit(x, y, n_jobs=n_jobs, use_neural_network=use_neural_network)

        # load trained models for prediction and scoring for testing data.
        # after we have fitted the trained models, then next step is to load whole of them from disk
        # and sort them based on score, and get scores for `test data` and `test label`
        # WE could get them from parent function, so that we could also use this for `regression`
        self.models_list = self._load_trained_models_ordered_by_score(higher_best=True)

        self._validation_models(xval, yval)

        logger.info("Whole training pipeline takes: {} seconds!".format(round(time.time() - start_time, 2)))

    @staticmethod
    def _get_data_and_label(file_load, x, y):
        """
        Ensure could get data and label.
        """
        if file_load is None and x is None and y is None:
            raise ValueError("When do real training, please provide at least a " +
                 "`file_load` or train data with `xtrain, ytrain`!")

        if file_load is not None:
            # with container, then just query the attribute then we could keep other as same.
            x, y = file_load.data, file_load.label
        else:
            if x is None or y is None:
                raise ValueError("When to do training, please provide both `x` and `y`!")
        
        return x, y
    
    def _validation_models(self, xval, yval):
        if xval is not None and yval is not None:
            score_dict = self.estimator.get_sorted_models_scores(xval, yval)
            print(score_dict)
            
            score_log_str = self.__format_trained_model_scores(score_dict)
            print(score_log_str)
        else:
            logger.warning("No need to validation!")
            pass

    @staticmethod
    def __format_trained_model_scores(score_dict, n_space=35):
        out_str_format = '{{0:{0}}}|{{1:{0}}}|{{2:{0}}}'.format(n_space)

        score_log_str = out_str_format.format("Model name", "Train score", "Validation score")
        logger.info(score_log_str)

        for model_name, test_score in score_dict.items():
            model_name_split = model_name.split('-')
            # model_name = model_name_split[0]
            train_score = model_name_split[1]
            train_score = train_score[:train_score.rindex(".")]
            # must convert to str, otherwise with not wanted result.
            test_score = str(test_score)
            
            log_str = out_str_format.format(model_name, train_score, test_score)
            score_log_str += '\n' + log_str

            logger.info(log_str)
        
        return score_log_str

    def get_sorted_models_scores(self, file_load=None, xtest=None, ytest=None, **kwargs):
        """
        To get some best trained model's score for `test` data with ordered.

        So that we could get the list of the best scores for later front end show case.
        :param x:
        :param y:
        :param kwargs:
        :return:
        """
        self._check_param(file_load, xtest, ytest)

        if file_load is not None:
            xtest, ytest = file_load.data, file_load.label

        score_dict = self.estimator.get_sorted_models_scores(xtest, ytest, reverse=True)

        return score_dict

    def predict(self, file_load=None, x=None, **kwargs):
        """To support with file_load obj with super func."""
        self._check_param(file_load, x)

        if file_load is not None:
            x, _ = self.__get_file_load_data_label(file_load)

        pred = super().predict(x)

        return pred

    def predict_proba(self, file_load=None, x=None, **kwargs):
        self._check_param(file_load, x)

        if file_load is not None:
            x, _ = self.__get_file_load_data_label(file_load)

        prob = super().predict_proba(x)

        return prob

    def score(self, file_load=None, x=None, y=None, **kwargs):
        self._check_param(file_load, x, y)

        if file_load is not None:
            x, y = self.__get_file_load_data_label(file_load, use_for_pred=False)
        
        score = super().score(x, y)

        return score

    @staticmethod
    def _check_param(*args):
        all_None = all([arg is None for arg in args])

        if all_None:
            raise ValueError("Please provide at least one parameter!")

    @staticmethod
    def __get_file_load_data_label(file_load, use_for_pred=True):
        """
        Get data and label for prediction data.
        """
        data, label = file_load.data, file_load.label

        if use_for_pred:
            if label is not None:
                raise ValueError("When to `predict` with `file_load` obj, have you set parameter: `use_for_pred=True`?" + 
                        "As we couldn't get label data from file_load obj.")
        
        return data, label

    @classmethod
    def reconstruct(cls, models_path=None, *args, **kwargs):
        """
        Used for Restful API to create object.
        """
        return cls(models_path, *args, **kwargs)

        

class FileLoad:
    """Load data from file, support with local file also with GCS.

    Make this class as a container for later use case.
    """
    def __init__(self, file_name, file_path=None, file_sep=',', label_name='label', use_for_pred=False,
            service_account_file_name=None, service_account_file_path=None, except_columns=None):
        """
        Support with prediction data load. If true, have to set parameter: `use_for_pred=True`
        """
        if not file_name.endswith('csv'):
            raise ValueError("Currently only support CSV file! Please provide with a CSV file.")
        self.file_name = file_name
        self.file_sep = file_sep
        self.file_path = file_path if file_path is not None else '.'
        self.label_name = label_name
        self.use_for_pred = use_for_pred
        self.service_account_file_name = service_account_file_name
        self.service_account_file_path = service_account_file_path
        if except_columns is not None and isinstance(except_columns, str):
            except_columns = [except_columns]
        self.except_columns = except_columns 
        self.data, self.label = self._load_data_file()

    def _get_file_location(self):
        if self.file_path is not None:
            if self.file_path.startswith("gs://"):
                return 'gcs'
            else:
                return 'local'

    def _get_gcs_file(self):
        # first service account
        self._check_service_account()

        bucket_name = self.file_path.split('/')[2]
        if bucket_name.find(".") > 0:
            # file_path also contain file name with `.`
            logger.error("When to get bucket name from file path: {}, couldn't get the bucket name.".format(self.file_path))
            raise ValueError("When to get bucket name from file path: {}, couldn't get the bucket name.".format(self.file_path))

        from google.cloud import storage
        
        # init client with service account file
        client = storage.Client.from_service_account_json(os.path.join(self.service_account_file_path, self.service_account_file_name))

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(self.file_name)

        try:
            # download as a file into temperate folder
            blob.download_to_filename(os.path.join(TMP_FOLDER, self.file_name))
            logger.info("File: {} has been downloaded from GCS to local.".format(self.file_name))

        except Exception as e:
            logger.error("Error downloading file with service account: {}".format(self.service_account_file_name))
            raise e
    
    def _check_service_account(self):
        # Check service account related
        if not self.service_account_file_name:
            raise ValueError("When try to use GCS, please must provide with service account file to interact with GCS!")

        if not self.service_account_file_name.endswith('json'):
            raise ValueError("When try to use GCS, Service acount file must be a JSON file, " +\
                 "but provided with {}".format(self.service_account_file_name))
                
        if not os.path.exists(os.path.join(self.service_account_file_path, self.service_account_file_name)):
            raise FileNotFoundError("When try to find service account file couldn't find file: {} in path: {}, " + 
                    "please check it.".format(self.service_account_file_name, self.service_account_file_path))
    
    def _load_data_file(self):
        file_location = self._get_file_location()
        
        file_path = self.file_path

        if file_location == 'gcs':
            logger.info("Start to download file:{} from GCS.".format(self.file_name))
            self._get_gcs_file()
            file_path = TMP_FOLDER 
            logger.info("Download file from GCS has finished.")
        
        try:
            df = pd.read_csv(os.path.join(file_path, self.file_name), sep=self.file_sep)

            df_cols = df.columns

            # In case that we don't need some columns and these cols should be in DF columns
            if self.except_columns is not None:
                except_common_cols = set(self.except_columns).intersection(list(df.columns))
                
                logger.info("Need to delete columns:{}  from original DataFrame.".format('\t'.join(list(except_common_cols))))

                if except_common_cols:
                    df.drop(except_common_cols, axis=1, inplace=True)

            if self.use_for_pred:
                # if just for prediction, then we don't get label data
                data = df.values
                label = None

                return data, label

            # let's inference data and label from DataFrame
           
            if self.label_name not in df_cols:
                # if `label` col not in the original file, then make let first column as `label` column
                label = df.iloc[:, 0]
                data = df.iloc[:, 1:]
            else:
                label = df[self.label_name]
                data = df.drop([self.label_name], axis=1)
            
            # convert into array type will be easier 
            data, label = data.values, label.values.reshape(-1, 1)
            
            return data, label
        except IOError as e:
            logger.error("Error load data file: {} from path: {}".format(self.file_name, file_path))
            raise e
        

if __name__ == '__main__':
    # Test with `FileLoad` class
    file_name = 'train.csv'
    file_path = r"C:\Users\guangqiiang.lu\Documents\lugq\code_for_future\auto_ml_pro\auto_ml\test"
    # file_path = "gs://cloud_sch_test"
    service_account_file_path = r"C:\Users\guangqiiang.lu\Downloads"
    service_account_name = "buoyant-sum-302208-4542dcd74629.json"

    # file_load = FileLoad(file_name, file_path, file_sep=',',  label_name='Survived', 
    #     service_account_file_name=service_account_name, service_account_file_path=service_account_file_path)
        
    file_load = FileLoad(file_name, file_path, file_sep=',',  label_name='Survived')
    models_path = r"C:\Users\guangqiiang.lu\Downloads\test_automl"

    auto_cl = ClassificationAutoML(models_path=models_path)

    # Start to train processing
    auto_cl.fit(file_load)
    print(auto_cl.models_list)
    # print(auto_cl.score(file_load_pred))

    file_load_pred = FileLoad("train.csv", file_path, label_name='Survived')
    print('*' * 20)
    print(auto_cl.predict(file_load_pred)[:10])
    print('*'*20)
    print(auto_cl.predict_proba(file_load_pred)[:10])

    # try to use sklearn iris dataset
    # from sklearn.datasets import load_iris
    # x, y = load_iris(return_X_y=True)
    # auto_cl.fit(x=x, y=y)

    # print(auto_cl.models_list)
    # print(auto_cl.score(xtest, ytest))
    # print('*' * 20)
    # print(auto_cl.predict(xtest)[:10])
    # print('*'*20)
    # print(auto_cl.predict_proba(xtest)[:10])

    # # get model score
    # print(auto_cl.get_sorted_models_scores(xtest, ytest))

    # This is used for the submition for Kaggle
    # pred_data = pd.read_csv(os.path.join(file_path, test_file_name))

    # pass_id_df = pred_data[['PassengerId']]

    # prediction = auto_cl.predict(file_load_pred)
    # pred_df = pd.DataFrame(prediction, columns=['Survived'])

    # pred_df = pd.concat([pass_id_df, pred_df], axis=1)

    # pred_df.to_csv(os.path.join(file_path,"Submition_with_automl.csv"), index=False)

