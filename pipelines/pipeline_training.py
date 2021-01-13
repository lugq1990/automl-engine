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
from sklearn.pipeline import Pipeline
from auto_ml.preprocessing import imputation
from auto_ml.utils.paths import load_yaml_file
from auto_ml.utils.backend_obj import Backend
from auto_ml.utils.logger import logger
from auto_ml.base.model_selection import GridSearchModel
from auto_ml.base.classifier_algorithms import ClassifierFactory
from auto_ml.preprocessing.processing_factory import ProcessingFactory
from auto_ml.ensemble.model_ensemble import ModelEnsemble


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
                 use_onehot=False,
                 use_standard=True,
                 use_norm=False,
                 use_pca=False,
                 use_minmax=False,
                 use_feature_seletion=False,
                 use_ensemble=True,
                 ensemble_alg='stacking',
                 voting_logic='soft'
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
        self.processing_pipeline = None
        self.training_pipeline = None
        self.algorithms_config = load_yaml_file()
        self.processor_config = load_yaml_file('default_processor.yml')['default']
        self.backend = Backend()
        # `ensemble` related
        self.use_ensemble = use_ensemble
        self.ensemble_alg = ensemble_alg
        self.voting_logic = voting_logic

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
        # TODO: But how to define the steps? Make it in the yaml file with order we want.

        # Here I add a pre-order that for some processing step must be added like `imputation`...
        # as some processing is a must, for other will be enhancement like feature-selection...
        # I add a logic here is: the most less important step should be first added, the most important
        # will be last inserted into 0 index... HERE add with data structure: Stack
        step_stack = []
        if self.use_feature_seletion or [True if data is not None and data.shape[1] > 20 else False][0]:
            step_stack.append('FeatureSelection')
        if self.use_pca or [True if data is not None and data.shape[1] > 20 else False][0]:
            step_stack.append('PrincipalComponentAnalysis')
        if self.use_minmax:
            step_stack.append('MinMax')
        if self.use_onehot:
            step_stack.append('OnehotEncoding')
        if self.use_standard:
            step_stack.append('Standard')
        if self.use_imputation:
            step_stack.append('Imputation')

        process_step = [step_stack.pop() for _ in range(len(step_stack))]

        # Whole need to add or delete processor steps should happen here.
        if self.include_estimators:
            process_step.extend([x for x in self.include_estimators if x not in process_step])

        if self.exclude_estimators:
            # not include some steps
            [process_step.remove(x) for x in self.exclude_estimators]

        # Here we should ensure that `process_step` should be at least 2 stages, otherwise will get error.
        if len(process_step) == 1:
            # add with `Standard` as most of algorithm would like data to be standard data.
            process_step.append('Standard')

        # return is a dictionary
        pipeline_steps = ProcessingFactory.get_processor_list(process_step)

        self.processing_pipeline = Pipeline(pipeline_steps)

        logger.info("Whole process pipeline steps: {}".format('\t'.join([x[0] for x in self.processing_pipeline.steps])))

        return self.processing_pipeline

    def build_training_pipeline(self):
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
        raise NotImplementedError

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

    def fit(self, x, y):
        """
        Real pipeline training steps happen here.
        :param x:
        :param y:
        :return:
        """
        start_time = time.time()
        logger.info("Start Model Training step!")

        self.training_pipeline = self.build_training_pipeline()

        logger.info("Before processing, data shape: %d" % x.shape[1])
        x_processed = self._fit_processing_pipeline(x, y)
        logger.info("After processing, data shape: %d" % x_processed.shape[1])

        try:
            # real training pipeline with Grid search to find best models, also will store the
            # best models.
            logger.info("Start to do pipeline training step.")
            self.training_pipeline.fit(x_processed, y)

            training_time = time.time() - start_time
            logger.info("Finished Pipeline training step, "
                        "whole training takes {} seconds.".format(round(training_time, 2)))

            # Whether or not to use `model_ensemble`
            if self.use_ensemble:
                logger.info("We are going to use `ensemble` logic to combine models")
                # so that we could config this based on what we want.
                kwargs = {"ensemble_alg": self.ensemble_alg, "voting_logic": self.voting_logic}
                self._fit_ensemble(x_processed, y, **kwargs)
                logger.info("`ensemble` training logic has finished.")

            return self
        except Exception as e:
            logger.error("When do real pipeline training get error: {}".format(e))
            raise Exception("When do real pipeline training get error: {}".format(e))

    def score(self, x, y):
        logger.info("Start to get accuracy score based on test data.")

        # Here to test that if we use the `load` processor to do processing logic
        processor = self.backend.load_model('processing_pipeline')
        x_processed = processor.transform(x)
        # x_processed = self.processing_pipeline.transform(x)

        acc_score = self.training_pipeline.score(x_processed, y)
        logger.info("Get accuracy score: %.4f" % acc_score)

        return acc_score

    def predict(self, x):
        try:
            logger.info("Start to get model prediction based on trained model")
            x_processed = self.processing_pipeline.transform(x)

            pred = self.training_pipeline.predict(x_processed)
            return pred
        except Exception as e:
            logger.error("When try to use pipeline to "
                            "get prediction with error: {}".format(e))
            raise Exception("When try to use pipeline to "
                            "get prediction with error: {}".format(e))

    def predict_proba(self, x):
        try:
            logger.info("Start to get model probability prediction based on trained model")

            prob = self.training_pipeline.predict_proba(x)

            return prob
        except Exception as e:
            logger.error("When try to use pipeline to get probability with error: {}".format(e))
            raise Exception("When try to use pipeline to "
                            "get probability with error: {}".format(e))

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

    @staticmethod
    def _fit_ensemble(data, label, **kwargs):
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

        model_ensemble = ModelEnsemble(ensemble_alg=ensemble_alg, voting_logic=voting_logic)

        try:
            # in fact with `training`, then the model will be saved into disk directly.
            # so that we don't need to care the rest, just `fit`
            model_ensemble.fit(data, label)
        except Exception as e:
            raise RuntimeError("When try to use `ModelEnsemble` to do model ensemble logic, "
                               "we get error: {}".format(e))


class ClassificationPipeline(PipelineTrain):
    """
    Classification pipeline class that we could use as a `pipeline`,
    also the `ensemble` logic should happen here.
    """
    def __init__(self):
        super(ClassificationPipeline, self).__init__()

    def build_training_pipeline(self):
        """
        Based on the `model_selection` module, when we do the fit logic,
        then that module will store the best models list into disk(This is
        based on the model parameters).
        So we don't need to care about the best model logic here.
        :return:
        """
        # This should be lazy part.
        self.training_pipeline = GridSearchModel()

        algorithms_instance_list = self._get_algorithms_instance_list()
        for algorithm_instance in algorithms_instance_list:
            self.training_pipeline.add_estimator(algorithm_instance)

        return self.training_pipeline

    def _get_algorithms_instance_list(self):
        """
        To get whole instance object list based on the configuration in the yaml file,
        as we have added with `factory pattern` in classifier class, so here we could
        just use the class to get whole algorithms instance.
        :return:
        """
        algorithm_name_list = self.algorithms_config['classification']['default']
        algorithms_instance_list = ClassifierFactory.get_algorithm_instance(algorithm_name_list)

        return algorithms_instance_list


class RegressionPipeline(PipelineTrain):
    def build_pipeline(self):
        """
        Here should be the regression step that could be used
        to build the pipeline object.
        :return:
        """
        pass


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)

    classifier_pipeline = ClassificationPipeline()
    # print(classifier_pipeline)

    # grid_models = classifier_pipeline.build_training_pipeline()
    # print(grid_models.list_estimators())
    # grid_models.fit(x, y)
    # print(grid_models.load_best_model_list())

    # process_pipeline = classifier_pipeline.build_preprocessing_pipeline()
    # print(process_pipeline)
    # classifier_pipeline._fit_processing_pipeline(x, y)

    classifier_pipeline.fit(x, y)
    print(classifier_pipeline.score(x, y))
