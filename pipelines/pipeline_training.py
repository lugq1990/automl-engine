# -*- coding:utf-8 -*-
"""
This class is used to do training for different algorithms.

This will just contain the training logic here with both preprocessing and algorithm training to
produce already trained models and dump them.

So if we do need to get the models' best score and prediction, the best process is to load
the trained model from disk and do `transformation` and `prediction`. If we need to do test,
then frist we need to do transformation based on the processor and use the highest score model
to do `prediction`. One important thing here: 1. save the processor; 2. dump trained data; 3.
dump whole trained models.

@author: Guangqiang.lu
"""
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tensorflow.keras import models

from auto_ml.utils.paths import load_yaml_file
from auto_ml.utils.backend_obj import Backend
from auto_ml.utils.logger import create_logger
from auto_ml.base.model_selection import GridSearchModel
from auto_ml.base.classifier_algorithms import ClassifierFactory
from auto_ml.base.regressor_algorithms import RegressorFactory
from auto_ml.preprocessing.processing_factory import ProcessingFactory
from auto_ml.ensemble.model_ensemble import ModelEnsemble
from auto_ml.utils.data_rela import get_scorer_based_on_target, get_type_problem
from auto_ml.utils.data_rela import check_data_and_label, hash_dataset_name



logger = create_logger(__file__)

class PipelineTrain(Pipeline):
    """
    Let's make it as parent class for both classification and regression.
    """
    def __init__(self,
                 include_estimators=None,
                 exclude_estimators=None,
                 include_preprocessors=None,
                 exclude_preprocessors=None,
                 use_imputation=True,
                 use_onehot=True,
                 use_standard=True,
                 use_norm=False,
                 use_pca=False,
                 use_minmax=False,
                 use_feature_seletion=False,
                 max_feature_num=20,
                 use_ensemble=True,
                 ensemble_alg='stacking',
                 voting_logic='soft',
                 backend=None,
                 ):
        self.include_estimators = include_estimators
        self.exclude_estimators = exclude_estimators
        self.include_preprocessors = include_preprocessors
        self.exclude_preprocessors = exclude_preprocessors
        self.use_imputation = use_imputation
        self.use_onehot = use_onehot
        self.use_standard = use_standard
        self.use_norm = use_norm
        self.use_pca = use_pca
        self.use_minmax = use_minmax
        self.use_feature_seletion = use_feature_seletion
        # How many features over our expect, then reduce the feature, default is 20
        self.max_feature_num = max_feature_num
        self.processing_pipeline = None
        self.training_pipeline = None
        self.algorithms_config = load_yaml_file()
        self.processor_config = load_yaml_file('default_processor.yml')['default']

        # self.backend = backend if backend is not None else Backend()
        self.backend = backend

        # `ensemble` related
        self.use_ensemble = use_ensemble
        self.ensemble_alg = ensemble_alg
        self.voting_logic = voting_logic

        # add `best_model` parameter for upper use case
        self.best_model = None

    def build_preprocessing_pipeline(self, data=None):
        """
        The reason that I want to split the preprocessing pipeline and training pipeline
        is that we will re-use the whole pre-processing steps if there contains some `null` values,
        so I think just to split the real pipeline into 2 parts: pre-processing and training.
        After whole steps finish, then I would love to store the processed data into disk,
        so that we could re-use the data.

        Also we need to store this pre-processing instance either combined with training pipeline instance.

        But I also want to add one more steps without the processing steps, as maybe the models could
        do better than with processing, we should store 3 parts data:
            1: origin data;
            2: data processed with imputation;
            3: data processed with whole processed.
        :param data:
        :return:
        """
        # except for included processor, then we also need to add some other's processor...

        process_step = self._get_process_step(data)

        # return is a dictionary
        pipeline_steps = ProcessingFactory.get_processor_list(process_step)

        self.processing_pipeline = Pipeline(pipeline_steps)

        if len(self.processing_pipeline.steps) == 0:
            raise RuntimeError("When try to build procesing pipeline, we couldn't get any steps, " + 
                        "please check whether you have provide a wrong parameter for `include_preprocessors` and `exclude_preprocessors`!")
                        
        logger.info("Whole process pipeline steps: {}".format('\t'.join([x[0] for x in self.processing_pipeline.steps])))

        return self.processing_pipeline

    def _get_process_step(self, data):
        """Get pre-define processing steps that we would take based on the dataset.

        Here I add a pre-order that for some processing step must be added like `imputation`...
        as some processing is a must, for other will be enhancement like feature-selection...
        I add a logic here is: the most less important step should be first added, the most important
        will be last inserted into 0 index... HERE add with data structure: Stack
        """
        step_stack = []
        if self.use_feature_seletion or \
                any([True if data is not None and data.shape[1] > self.max_feature_num else False]):
            step_stack.append('FeatureSelection')
        if self.use_pca or \
                any([True if data is not None and data.shape[1] > self.max_feature_num else False]):
            step_stack.append('PrincipalComponentAnalysis')
        if self.use_minmax:
            step_stack.append('MinMax')
        if self.use_standard:
            step_stack.append('Standard')
        if self.use_onehot:
            # One-hot logic should happen after the imputation logic! As we need numeric data.
            step_stack.append('OnehotEncoding')
        if self.use_imputation:
            # Add a logic that if there isn't any NAN value, just pass this
            if np.isnan(data).any():    
                step_stack.append('Imputation')

        process_step = [step_stack.pop() for _ in range(len(step_stack))]

        # Pre-configured processor that we would like or don't need.
        if self.include_preprocessors:
            process_step.extend([x for x in self.include_preprocessors if x not in process_step])

        if self.exclude_preprocessors:
            # not include some steps
            [process_step.remove(x) for x in self.exclude_preprocessors]

        # Here we should ensure that `process_step` should be at least 2 stages, otherwise will get error.
        if len(process_step) == 1:
            # add with `Standard` as most of algorithm would like data to be standard data.
            process_step.append('Standard')

        return process_step

    def build_training_pipeline(self, y=None, use_neural_network=True):
        """
        Real pipeline step should happen here.
        Let child to do real build with different steps
        and add the steps instance into `pipeline` object.
        Also I think here should a lazy instant step, should happen
        when we do real fit logic, so that we could also based on data to modify our steps.

        Important thing to notice:
            I think even we have many algorithm instances, first step should combine processing step
            with each algorithm, then we could get some best scores models and save them into disk.

            Then we could load them from disk and combine them with `ensemble logic`!

            I have created a model_selection module to get best models based on training data,
            so here don't need a list of pipeline objects.
        :return: a list of instance algorithm object.
        """
        # This should be lazy part.
        self.training_pipeline = GridSearchModel(backend=self.backend, use_neural_network=use_neural_network, task_type=self.task_type)

        # This is based on diff type of problem to get each algorithm's instance! 
        # Tips: Parent class could call Child private function!
        algorithms_instance_list = self._get_algorithms_instance_list(y=y)

        print("Get algorithms: ", algorithms_instance_list)
        logger.info("Get training algorithms: {}".format([al_instance.name for al_instance in algorithms_instance_list]))

        for algorithm_instance in algorithms_instance_list:
            self.training_pipeline.add_estimator(algorithm_instance)

        return self.training_pipeline

    def _fit_processing_pipeline(self, x, y=None):
        """
        To split the pre-processing pipeline here.
        :param x:
        :param y:
        :return:
        """
        self.processing_pipeline = self.build_preprocessing_pipeline(x)

        # processing pipeline with training data
        logger.info("Start to do pre-processing step.")
        self.processing_pipeline.fit(x, y)

        # Before we do real training pipeline, we should first do the data transformation and store data and
        # processing object into disk
        logger.info("Start to transform training data.")
        x_processed = self.processing_pipeline.transform(x)

        # To save the trained model and transformed dataset into disk.
        logger.info("Start to save the processor object and processed data into disk.")

        self.backend.save_model(self.processing_pipeline, 'processing_pipeline')

        # This is to save processed data into disk, so should be in tmp folder.
        logger.info("Start to save processed data into disk!")
        self.backend.save_dataset(x_processed, 'processed_data', model_file_path=False)

        return x_processed

    def fit(self, x, y, n_jobs=None, use_neural_network=True):
        """
        Real pipeline training steps happen here.
        :param x:
        :param y:
        :return:
        """
        logger.info("Start Model Pipeline training!")

        # self.task_type = get_type_problem(y)
        self.training_pipeline = self.build_training_pipeline(y=y, use_neural_network=use_neural_network)

        logger.info("Before processing, data shape: %d" % x.shape[1])
        # first should `fit` with processing pipeline and save it into disk.
        x = self._fit_processing_pipeline(x, y)
        logger.info("After processing, data shape: %d" % x.shape[1])

        self._fit(x, y, n_jobs=n_jobs)
        logger.info("End Model Pipeline training.")

        # After model has finished training, the best_model should be `training_pipeline`
        self.best_model = self.training_pipeline

        return self

    def score(self, x, y):
        """
        Default will just use the best trained estimator to do score.
        :param x:
        :param y:
        :return:
        """
        # func `get_scorer_based_on_target` already with type of problem
        # should based on the prediction not estimator
        scorer = get_scorer_based_on_target(self.task_type)

        score = self._get_model_score(self.training_pipeline, x, y, scorer)

        return score

    def predict(self, x):
        """
        Based on the `training_pipeline` to get prediction
        :param x:
        :return:
        """
        pred = self._get_model_predict(self.training_pipeline, x)

        return pred

    def predict_proba(self, x):
        """
        Based on the `training_pipeline` to get probability
        :param x:
        :return:
        """
        prob = self._get_model_predict_proba(estimator=self.training_pipeline, x=x)

        return prob

    def get_sorted_models_scores(self, x, y, reverse=True):
        """
        Add this func to get whole score based for each trained models, so that
        we could get the result that we have taken that times and for each models,
        how about the testing result.

        Load `whole trained models` from disk and do processing for the new data, and
        score based on each model with different type of problem.
        :param x:
        :param y:
        :param reverse:
            Whether or not to order the result based the `reverse`.
        :return:
            sorted dictionary: {'lr-0.982': 0.87, ...}
        """
        score_dict = {}

        # load whole trained models from disk, [('LR-0.98.pkl', LogisticRegression-0.982323.pkl)]
        models_list = self.backend.load_models_combined_with_model_name()

        logger.info("Get model list: " + '\t'.join([model[0] for model in models_list]))
        if not models_list:
            logger.warning("There isn't any trained model loaded from model path!")
            return None

        scorer = get_scorer_based_on_target(self.task_type)

        # first should process this dataset with trained processor, so later step will be easier
        # x = self._process_dataset_with_processor(x)

        # `models_list` also contain `ensemble model`,
        # so if we get `stacking`, we should make new dataset

        for model_tuple in models_list:
            model_name, model_instance = model_tuple[0], model_tuple[1]

            if model_name.lower().startswith('stacking'):
                logger.info("Get `Ensemble` processing data to get score for testing data.")
                score = self._get_model_score(model_instance, x, y, scorer, model_name=model_name)
            else:
                score = self._get_model_score(model_instance, x, y, scorer)

            score = round(score, 6)

            score_dict[model_name] = score

        if not score_dict:
            logger.error("When to get model score, we haven't get any score, please check current func:`get_sorted_models_scores`!")
            raise ValueError("When to get model score, we haven't get any score, please check current func:`get_sorted_models_scores`!")

        # based on parameter: `reverse`
        score_dict ={k: v for k, v in sorted(score_dict.items(),  key=lambda x: x[1], reverse=reverse)}

        return score_dict

    def _get_model_score(self, estimator, x, y, scorer, model_name=None):
        """
        Use for get each `trained` model's score.
        :param estimator:
        :param x:
        :param y:
        :param scorer:
            scorer needed to be provided, as func will be called many times.
        :return:
        """
        logger.info("Start to get model score for model: {}.".format(estimator))
        pred = self._get_model_predict(estimator, x, model_name=model_name)

        # This is supported with both regression and classification, 
        # as `scorer` is a class instance based on each type of problem.
        score = scorer(y, pred)
        logger.info("Model: {} get score: {:.4f}.".format(estimator, score))

        return score

    def _get_model_predict(self, estimator, x, model_name=None):
        """
        Prediction func should based on best trained model score instance.
        
        :param x: data should be processed already!
        :return:
        """
        prob = self._get_model_predict_proba(estimator, x, model_name)
        if self.task_type == 'classification':
            if len(prob) >= 1:
                pred = np.argmax(prob, axis=1)
                
                return pred
        else:
            pred = prob

        if len(pred.shape) == 1:
            pred = pred.reshape(-1, 1)
        
        return pred

    def _get_model_predict_proba(self, estimator, x, model_name=None):
        """
        Probability func should based on this func.
        :param estimator:
        :param x:
        :return:
        """
        if estimator is None:
            # This is used for on-line prediction that we couldn't get training status that we have already get.
            # so here is try to re-construct best trained model from disk.
            logger.warning("When to get model prediction, estimator hasn't been provided, "
                            "use best trained model from disk to get prediction!")
            # if the estimator is None, then try to get the best score instance for this.
            models_list = self.backend.load_models_combined_with_model_name()

            if len(models_list) == 0:
                logger.error("When try to get model prediction, couldn't get trained models from disk! Please try to check!")
                return None

            try:
                # This is changed as the score is with `AdaboostClassifier_0.766667.pkl`
                model_score = []
                for ml_name, _ in models_list:
                    ml_score = ml_name.split('_')[1]
                    ml_score = float(ml_score[:ml_score.rindex('.')])
                    model_score.append(ml_score)
                # model_score = [float(model_name.split('-')[1].split('.')[0]) for model_name, _ in models_list]
                estimator = models_list[np.argmax(model_score)][1]    

                if model_name is None:
                    model_name = models_list[np.argmax(model_score)][0]

                logger.info("Get best estimator: {}".format(model_name))
            except ValueError as e:
                logger.error("When try to get model score based on trained model name, get error: {}".format(e))
                raise e

        if self.task_type == 'classification':   
            if not hasattr(estimator, 'predict_proba'):
                raise NotImplementedError("For estimator:{} "
                                            "doesn't support `predict_proba` func".format(self.training_pipeline))

        # let's just make the processing step with data here!
        x = self._process_dataset_with_processor(x)

        # When we do real processing for `stacking`, new created dataset should happen here.
        if model_name is not None:
            if model_name.lower().startswith('stacking'):
                logger.info("Start to produce new dataset with `stacking` class.")
                x = ModelEnsemble.create_stacking_dataset(x, backend=self.backend, task_type=self.task_type)
            else:
                raise ValueError("When to model prediction, model name: {} is not supported!".format(model_name))

        if self.task_type == 'classification':
            prob = estimator.predict_proba(x)
        else:
            prob = estimator.predict(x)

        return prob
        
    def _fit(self, x, y, n_jobs=None):
        """
        Extract real `fit` logic out side of `fit` func.
        :param data:
        :return:
        """
        # before we do anything, first should ensure we have proper data and label
        self._check_data_and_label(x, y)

        # real training pipeline with Grid search to find best models, also will store the
        # best models.
        logger.info("Start to do pipeline training step.")
        start_time = time.time()

        self.training_pipeline.fit(x, y, n_jobs=None)
        logger.info("Training pipeline finished takes: {} seconds.".format(round((time.time() - start_time)), 2))

            # Whether or not to use `model_ensemble`
        if self.use_ensemble:
            logger.info("We are going to use `ensemble` logic to combine models!")
            # so that we could config this based on what we want.
            kwargs = {"ensemble_alg": self.ensemble_alg, "voting_logic": self.voting_logic}

            start_time = time.time()
            self._fit_ensemble(x, y, **kwargs)
            logger.info("`Ensemble` training pipeline finished takes: {} seconds.".format(round(time.time() - start_time), 2))

    def __repr__(self):
        if self.training_pipeline is None:
            return "Pipeline hasn't been fitted, as this is lazy instance, so after fitted step, then we could see " \
                   "whole steps!"
        else:
            print(self.training_pipeline)
            steps_str = ""
            for step in self.processing_pipeline.steps:
                steps_str += step[0] + '\n'
            for step in self.training_pipeline.steps:
                steps_str += step[0] + '\n'

            return steps_str

    def _fit_ensemble(self, data, label, **kwargs):
        """
        Based on trained models to do model ensemble logic to try to get better model.

        For `fitting`, we could provide key-words like: `ensemble_alg` and `voting_logic` etc.
        to init our model
        :param data:
        :param label:
        :return:
        """
        if kwargs:
            ensemble_alg = kwargs['ensemble_alg']
            voting_logic = kwargs['voting_logic']
        else:
            ensemble_alg = 'stacking'
            voting_logic = 'soft'

        # for checking which type of problem
        model_ensemble = ModelEnsemble(backend=self.backend, 
            ensemble_alg=ensemble_alg, 
            voting_logic=voting_logic, 
            task_type=self.task_type)

        # in fact with `training`, then the model will be saved into disk directly.
        # so that we don't need to care the rest, just `fit`
        logger.info("Start to build `ensemble` logic using `{}` algorithm with voting logic: {}".format(ensemble_alg, voting_logic))
        
        model_ensemble.fit(data, label)
        

    def _process_dataset_with_processor(self, x):
        """
        To process the data with trained processor, this is used for `pipeline` processing data.

        :param x:
        :return: processed data with processor object.
        """
        processor = self.backend.load_model("processing_pipeline.pkl")

        try:
            x_processed = processor.transform(x)

            return x_processed
        except Exception as e:
            logger.error("When to process data in `ensemble`, get error: {}".format(e))
            raise e

    def _check_data_and_label(self, x, y):
        """
        Let child to do this.
        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError

    def _get_algorithms_instance_list(self, y=None):
        """
        Get algorithm instance by factory class based on target data.

        To get whole instance object list based on the configuration in the yaml file,
        as we have added with `factory pattern` in classifier class, so here we could
        just use the class to get whole algorithms instance.

        Noted: Should support with classification and regression.
        :return:
        """
        # If `included_estimator`, then only get these algorithm instance
        algorithm_name_list = []
        if self.include_estimators:
            if isinstance(self.include_estimators, str):
                algorithm_name_list = [self.include_estimators]
            elif isinstance(self.include_estimators, list):
                algorithm_name_list = self.include_estimators
            else:
                raise ValueError("Please config `include_estimators` with str or a list of algorithm names!")
        else:
            # then just get default algorithms based on target value.
            # task_type = get_type_problem(y)

            print("**** get type of problem ****", self.task_type)

            algorithm_name_list = self.algorithms_config[self.task_type]['default']
            
            # When to get instances, should base on each type of problem
            if self.task_type == 'classification':
                algorithms_instance_list = ClassifierFactory.get_algorithm_instance(algorithm_name_list)
            else:
                algorithms_instance_list = RegressorFactory.get_algorithm_instance(algorithm_name_list)

        return algorithms_instance_list


class ClassificationPipeline(PipelineTrain):
    """
    Classification pipeline class that we could use as a `pipeline`,
    also the `ensemble` logic should happen here.
    """
    def __init__(self, backend=None, 
                    include_estimators=None, 
                    exclude_estimators=None,
                    include_preprocessors=None,
                    exclude_preprocessors=None,
                    **kwargs):
        super(ClassificationPipeline, self).__init__(backend=backend, 
                                            include_estimators=include_estimators,
                                            exclude_estimators=exclude_estimators,
                                            include_preprocessors=include_preprocessors,
                                            exclude_preprocessors=exclude_preprocessors,
                                            **kwargs)
        self.task_type = 'classification'

    def _check_data_and_label(self, x, y, task=None):
        """
        Check data and label before we do real training.
        :param x:
        :param y:
        :param task:
        :param dataset_name:
        :return:
        """
        # first we should get the task type, that's for what metrics to use.
        classification_tasks = [0, 1, 2]
        if task is None:
            # we should get the task type by self.
            task = self._get_task_type(y)
        if task not in classification_tasks:
            raise ValueError("Task should be in [{}]".format(' '.join(classification_tasks)))

        # self.dataset_name = hash_dataset_name(x) if dataset_name is None else dataset_name
        # we should ensure data could be trained like training data should be 2D
        x, y = check_data_and_label(x, y)

        return x, y

    @staticmethod
    def _get_task_type(y):
        """
        Get the target type that we want to do.
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


class RegressionPipeline(PipelineTrain):
    def __init__(self, backend=None, 
                    include_estimators=None, 
                    exclude_estimators=None,
                    include_preprocessors=None,
                    exclude_preprocessors=None,
                    **kwargs):
        super(RegressionPipeline, self).__init__(backend=backend, 
                                            include_estimators=include_estimators,
                                            exclude_estimators=exclude_estimators,
                                            include_preprocessors=include_preprocessors,
                                            exclude_preprocessors=exclude_preprocessors,
                                            **kwargs)
        self.task_type = 'regression'

    def _check_data_and_label(self, x, y, task=None):
        """Data and label should be the one that we need for regression.

        Args:
            x ([type]): [description]
            y ([type]): [description]
            task ([type], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        regression_tasks = [3]
        
        if task is not None and task not in regression_tasks:
            raise ValueError("Task should be in [{}]".format(' '.join(regression_tasks)))

        # self.dataset_name = hash_dataset_name(x) if dataset_name is None else dataset_name
        # we should ensure data could be trained like training data should be 2D
        x, y = check_data_and_label(x, y)

        return x, y


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)

    from auto_ml.utils.backend_obj import Backend

    models_path = r"C:\Users\guangqiiang.lu\Downloads\test_automl"
    backend = Backend(output_folder=models_path)
    print("Backend output folder ", backend.output_folder)

    classifier_pipeline = ClassificationPipeline(backend=backend)
    # print(classifier_pipeline)

    # grid_models = classifier_pipeline.build_training_pipeline()
    # print(grid_models.list_estimators())
    # grid_models.fit(x, y)
    # print(grid_models.load_best_model_list())

    # process_pipeline = classifier_pipeline.build_preprocessing_pipeline()
    # print(process_pipeline)
    # classifier_pipeline._fit_processing_pipeline(x, y)

    from auto_ml.test.get_test_data import get_training_data
    from sklearn.model_selection import train_test_split

    # x, y = get_training_data()
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)

    classifier_pipeline.fit(xtrain, ytrain)
    pred = classifier_pipeline.predict(xtrain)

    print("Model score: ", classifier_pipeline.score(xtest, ytest))

    print(classifier_pipeline.get_sorted_models_scores(xtest, ytest))
