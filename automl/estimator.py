# -*- coding:utf-8 -*-
"""
Entry point for full `automl` training pipeline with support for `classfication` and `regression`.

Support with base machine learning models training and nueral network training, not only with just
one model training, but also with `ensemble` to combine trained models into a more robust model to 
both reduce `variance` and `bias`. 

High level steps:

1. Load training and testing data file or memory objects.

2. Feature engineering step to process data.

3. Model training based on processed data.

4. Nueral network model training based on processed data.

5. Ensemble logic to combine trained model and do comparation to see better or not.

6. Dump trained models into disk with user defined path.

author: Guangqiang.lu
"""
from __future__ import absolute_import

import pandas as pd
import time
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from automl.utils.backend_obj import Backend
from automl.classifier_algorithms import *
from automl.scorer import *
from automl.utils.CONSTANT import *
from automl.pipeline_training import ClassificationPipeline,  RegressionPipeline
from automl.utils.logger import create_logger


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
        Parent class for both classificatinon and regression auto training class.
        
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

    def fit(self, x=None, y=None, file_load=None, 
             xval=None, yval=None, val_split=.2, 
             n_jobs=None, use_neural_network=True, *args, **kwargs):
        """Main training entry point with support with file and memory objects.
        
        Full training step with pre-processing pipeline and training pipeline happens here.
        Various type of data is supported and will convert them into a normal array for later
        training algorithms, will instant a training pipeline with different algorithms with
        hyper-parameters selected, will use grid-search to find best hyper-parameters,  will 
        store these trained models with validation score attached with algorithm name.

        Args:
            x ([array], optional): [training data]. Defaults to None.
            y ([array], optional): [training label]. Defaults to None.
            file_load ([FileLoad], optional): [file_load object to contain data and label]. Defaults to None.
            xval ([array], optional): [validation data]. Defaults to None.
            yval ([array], optional): [validation label]. Defaults to None.
            val_split ([float], optional): [percentage for validation if `xval` and `yval` not provdied]. 
                Defaults to 0.2.
            n_jobs ([int], optional): [how many cores to be used]. Defaults to None.
            use_neural_network (bool, optional): [whether or not to use neural networks.]. Defaults to True.

        Returns:
            [self]: [trained object.]
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

        # After change regression metrics to r2, this is workable
        self.models_list = self._load_trained_models_ordered_by_score(higher_best=True)

        self._validation_models(xval, yval)

        logger.info("Whole training pipeline takes: {} seconds!".format(round(time.time() - start_time, 2)))

        return self

    def predict(self, x=None, file_load=None, **kwargs):
        """Based on data or file to get prediction based on best trained models.

        Args:
            x ([array], optional): [test data]. Defaults to None.
            file_load ([FileLoad], optional): [file_load object to contain data and label]. Defaults to None.

        Returns:
            [array]: [prediction]
        """
        x = self._get_training_data(x, file_load)

        logger.info("Start to get prediction based on best trained model.")
        prediction = self.estimator.predict(x)
        logger.info("Prediction step finishes.")

        return prediction

    def predict_proba(self, x=None, file_load=None, **kwargs):
        """Probability supported based on best trained model.

        Args:
            x (array, optional): test data. Defaults to None.
            file_load (array, optional): file_load object. Defaults to None.

        Raises:
            NotImplementedError: Raise error if not support with `predict_proba`

        Returns:
            array: probability of test data
        """
        # should check the estimator should have function: `predict_proba`!
        if not hasattr(self.estimator, 'predict_proba'):
            raise NotImplementedError("Best fitted model:{} doesn't support `predict_proba`".format(self.best_model))

        x = self._get_training_data(x, file_load)

        logger.info("Start to get probability based on best trained model.")
        prob = self.estimator.predict_proba(x)
        logger.info("Prediction step finishes.")

        return prob

    def score(self, x=None, y=None, file_load=None, **kwargs):
        """Get score based on test data and label.
        
        Classifcation will use `accuracy`, regression will use `r2-score`

        Args:
            x (array, optional): test data. Defaults to None.
            y (array, optional): test label. Defaults to None.
            file_load (FileLoad, optional): file_load to contain data and label. Defaults to None.

        Returns:
            float: evaluation score
        """
        self._check_fitted()

        logger.info("Start to get prediction based on best trained model!")

        self._check_param(file_load, x, y)

        if file_load is not None:
            x, y = self.__get_file_load_data_label(file_load, use_for_pred=False)

        # ensure we have array type
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
            
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

        # Use child func, child should implement with score based on different type of problem.
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
                             key=lambda model: float(model[0].split("_")[1].replace(".pkl", '').replace(".h5", '')),
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
        """Check is trained or not?

        Raises:
            NotFittedError: If there isn't any models_list object, raise error

        Returns:
            boolean: True if does have models_list
        """
        if not self.models_list:
            logger.error("When try to get prediction with `automl`, Please fit the model first")
            raise NotFittedError("When try to get prediction with `automl`, Please fit the model first")

        return True

    @staticmethod
    def _check_param(*args):
        """Check parameters are all None.

        Raises:
            ValueError: Full parameters are None.
        """
        all_None = all([arg is None for arg in args])

        if all_None:
            raise ValueError("Please provide at least one parameter!")

    @staticmethod
    def __get_file_load_data_label(file_load):
        """Get data and label from original obj.

        Args:
            file_load (FileLoad): Container object for data

        Returns:
            tuple: (data, label)
        """
        data, label = file_load.data, file_load.label
        
        return data, label

    @classmethod
    def reconstruct(cls, models_path=None, *args, **kwargs):
        """Used for Restful API to create

        Args:
            models_path (str, optional): Where trained model is. Defaults to None.

        Returns:
            AutoML: a re-constructed object for API use case
        """
        return cls(models_path, *args, **kwargs)

    @staticmethod
    def _get_data_and_label(file_load, x, y):
        """Get data and label with file_load or just data and label provided.

        Args:
            file_load (FileLoad): Container
            x (array): data
            y (array): label

        Raises:
            ValueError: Nothing provided, full is None
            ValueError: Couldn't get data and label

        Returns:
            tuple: data and label
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

        # Let's try to make DF into array, so later will be easier!
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        
        return x, y
    
    def _validation_models(self, xval, yval):
        """Print out validation based on full trained models with score.

        Args:
            xval (array): validation data.
            yval (array): validation label.
        """
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
        """Provide a format for output.

        Args:
            score_dict (dict): model_name with score
            n_space (int, optional): how many spaces to be used. Defaults to 35.

        Returns:
            str: string format
        """
        out_str_format = '{{0:{0}}}|{{1:{0}}}|{{2:{0}}}'.format(n_space)

        score_log_str = out_str_format.format("Model name", "Train score", "Validation score")
        logger.info(score_log_str)

        for model_name, test_score in score_dict.items():
            try:
                model_name_split = model_name.split('_')
                # model_name = model_name_split[0]
                train_score = model_name_split[1]
                train_score = train_score[:train_score.rindex(".")]
                # must convert to str, otherwise with not wanted result.
                test_score = str(test_score)
                
                log_str = out_str_format.format(model_name, train_score, test_score)
                score_log_str += '\n' + log_str
            except Exception as e:
                logger.warning("Get invalidate model name: {}".format(model_name))
                continue

            logger.info(log_str)
        
        return score_log_str

    def get_sorted_models_scores(self, xtest=None, ytest=None, file_load=None, reverse=True, **kwargs):
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

        score_dict = self.estimator.get_sorted_models_scores(xtest, ytest, reverse=reverse)

        return score_dict

    def _get_training_data(self, x, file_load):
        """Get training data based on `file_load` or `x`.
        
        if x is provided, then just return x, otherwise should get data 
        from file_load.

        Either of thems should be provided.

        Args:
            x ([type]): [description]
            file_load ([type]): [description]

        Returns:
            [type]: [description]
        """
        self._check_param(file_load, x)

        if x is not None:
            if isinstance(x, pd.DataFrame):
                x = x.values
            return x
        elif file_load is not None:
            x, _ = self.__get_file_load_data_label(file_load)   
            return x


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
        

class RegressionAutoML(AutoML):
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
        super(RegressionAutoML, self).__init__(models_path=models_path,
                                            include_estimators=include_estimators,
                                            exclude_estimators=exclude_estimators,
                                            include_preprocessors=include_preprocessors,
                                            exclude_preprocessors=exclude_preprocessors, **kwargs)        

        # after pipeline has finished, then we should use `ensemble` to combine these models
        # action should happen here.
        self.estimator = RegressionPipeline(backend=self.backend,
                                            include_estimators=include_estimators,
                                            exclude_estimators=exclude_estimators,
                                            include_preprocessors=include_preprocessors,
                                            exclude_preprocessors=exclude_preprocessors, 
                                            **kwargs)

class FileLoad:
    """Load data from file, support with local file also with GCS.

    Make this class as a container for later use case.
    """
    def __init__(self, file_name, file_path=None, file_sep=',', label_name='label', use_for_pred=False,
            service_account_file_name=None, service_account_file_path=None, except_columns=None):
        """Main container for file-like dataset.

        Args:
            file_name (str): Name of file
            label_name (str, optional): What is `label` column's name?. Defaults to 'label'.
            file_path (str, optional): Where file located?. Defaults to None.
            file_sep (str, optional): File seprator. Defaults to ','.
            use_for_pred (Boolean, optional): Whether to use this for prediction? 
                Noted: If file doesn't contain label column, do need set this parameter to `True`. Defaults to False.
            service_account_file_name (str, optional): SA file name. Defaults to None.
            service_account_file_path (str, optional): SA file path. Defaults to None.
            except_columns (List, optional): Columns are needed to be used. Defaults to None.

        Raises:
            ValueError: [description]
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

        self.columns = None

    def _get_file_location(self):
        if self.file_path is not None:
            if self.file_path.startswith("gs://"):
                return 'gcs'
            else:
                return 'local'

    def _get_gcs_file(self):
        """Download the GCS file into local tmp folder.

        Raises:
            ValueError: Couldn't get bucket path.
            e: Download fail with service account.
        """
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
        """Check service account is exist and with JSON format.

        Raises:
            ValueError: Service account isn't provided.
            ValueError: Service account isn't with JSON extension.
            FileNotFoundError: Service account file isn't exist.
        """
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
        """Load data and label based on file_load object.

        Raises:
            e: Load fail

        Returns:
            tuple: data and label
        """
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
            self.columns = df_cols

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
    from automl.estimator import ClassificationAutoML, FileLoad, RegressionAutoML
    
    file_name = 'train.csv'
    file_path = r"C:\Users\guangqiang.lu\Documents\lugq\github\auto-ml-cl\automl\test"
        
    file_load = FileLoad(file_name, file_path, file_sep=',',  label_name='Survived')
    models_path = r"C:\Users\guangqiang.lu\Downloads\test_automl"

    auto_est = ClassificationAutoML(models_path=models_path)
    # auto_est = RegressionAutoML(models_path=models_path)

    # try to use sklearn iris dataset
    from sklearn.datasets import load_boston, load_iris
    from sklearn.model_selection import train_test_split
    
    x, y = load_iris(return_X_y=True)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)
    
    auto_est.fit(xtrain, ytrain)

    # regression_file_path = r"C:\Users\guangqiang.lu\Documents\lugq\Kaggle\HousePricePredict"
    # file_load = FileLoad('train.csv', file_path=regression_file_path, label_name='SalePrice')
    # auto_est.fit(file_load=file_load)

    print(auto_est.models_list)
    print(auto_est.score(xtest, ytest))
    print('*' * 20)
    print(auto_est.predict(xtest)[:10])
    print('*' * 20)
    pred = auto_est.predict(xtest)
    print(auto_est.predict_proba(xtest)[:10])
    print("Truth data: ")
    print(ytest)

