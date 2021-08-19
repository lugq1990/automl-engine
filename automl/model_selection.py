# -*- coding:utf-8 -*-
"""
Main Cross-validation class for adding more estimators and get score using CV.

This will contain some model selection logic should be used here like
Grid search logic, as here what I could do is create the search space.
So maybe Grid search to find whole models.

Classifier real training happens here.

@author: Guangqiang.lu
"""
from operator import ne
import numpy as np
import time
import tqdm
import itertools
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import metrics

from .utils.backend_obj import Backend
from .utils.logger import create_logger
from .utils.data_rela import get_type_problem, get_scorer_based_on_target
from .neural_network_search import NeuralModelSearch, default_neural_algorithm_list


logger = create_logger(__file__)


class GridSearchModel(object):
    """
    Here could just to implement that could add a list
    of estimators and their parameters list.
    I want to make this class to do real training part.
    """
    def __init__(self, backend, n_best_model=None, use_neural_network=True, task_type='classification'):
        """
        self.estimator_list is like: [GridSearchCV(lr, params), ...]
        self.score_dict is like: {'LogisticRegressin': (lr, 0.9877)}
        """
        super(GridSearchModel, self).__init__()
        # as we need to do training, so here will just store the trained best model
        # for later step ensemble
        if backend is None:
            raise ValueError("When to use Model Ensemble class, we get a None `backend` object! Please check!")
        self.backend = backend 
        self.use_neural_network = use_neural_network
        # estimator_list and score_list are binded! they are same order!
        self.estimator_list = []
        self.score_list = []
        self._estimator_param_list = []
        self.n_best_model = 30 if n_best_model is None else n_best_model
        self.task_type = task_type

        self.best_estimator = None
        self.best_score = 0

    def add_estimator(self, estimator, estimator_params=None):
        """
        As I also want to keep current logic with a list
        of estimators in parallel, so here should keep
        whole models.
        As I will my own estimator, so don't need always need
        `estimator_params`, but just add in case we just want
        to add other sklearn estimator
        :param estimator: estimator object
        :param estimator_params: native sklearn object params to search.
        :return:
        """
        if hasattr(estimator, 'get_search_space'):
            # so that this is our estimator
            if len(self._estimator_param_list) == 0:
                self._estimator_param_list = [(estimator, estimator.get_search_space())]
            else:
                self._estimator_param_list.append((estimator, estimator.get_search_space()))
        else:
            if estimator_params is None:
                raise ValueError("When we need to set other sklearn native estimator, do"
                                 "need to add estimator_params for searching")

            if len(self._estimator_param_list) == 0:
                self._estimator_param_list = [(estimator, estimator_params)]
            else:
                self._estimator_param_list.append((estimator, estimator_params))

    def add_estimators(self, estimator_param_pairs):
        """
        This is try to parallel training for whole training
        with different params.
        :param estimator_param_pairs: a list of estimator and params
         just like this:[(lr, {"C":[1, 2]})]
        :return:
        """
        for estimator, estimator_params in estimator_param_pairs:
            self.add_estimator(estimator, estimator_params)

    def _get_estimators_list(self):
        """Build a estimator list that with each parameter setting.

        As there maybe many keys that needed to be set, so we need make a `product` of whole keys.
        """
        estimator_list = []

        for estimator, estimator_param in self._estimator_param_list:
            param_keys = list(estimator_param.keys())
            param_values = []
            for k in param_keys:
                param_values.append(estimator_param[k])
            param_product = list(itertools.product(*param_values))

            for i in range(len(param_product)):
                estimator_params = {}
                for j in range(len(param_keys)):
                    estimator_params[param_keys[j]] = param_product[i][j]
                    estimator.set_params(**estimator_params)
                    
                    estimator_list.append(estimator)

        return estimator_list

    def fit(self, x, y, n_jobs=None):
        """
        Fit and Cross-validation for each estimator with `scoring` supported!
        
        After CV and re-fit with full data, also the `training_score` and `testing_score` is based on CV result. 
        Support with processing information by `tqdm`.

        By defualt: 
            `classification` scoring is `accuracy`
            `regression` scoring is `mean_squared_error`
    
        :param x: training data
        :param y: training label
        :param n_jobs: how much cores to use
        :return:
        """
        # parallel training
        if n_jobs is not None and n_jobs > 1:
            # here couldn't use multiprocessing here, just to set
            # estimator `n_job`
            # we could add other multiprocessing here either if we want.
            logger.info("Start to train model based on {} cores.".format(n_jobs))
            # set `n_jobs` for each estimator
            for i in range(len(self._estimator_param_list)):
                estimator = self._estimator_param_list[i][0]
                if hasattr(estimator, 'n_jobs'):
                    estimator.n_jobs = n_jobs
                    self._estimator_param_list[i][0] = estimator

        estimators_list = self._get_estimators_list()

        # Get scoring metrics based on different type of problem.
        scorer = get_scorer_based_on_target(self.task_type)
        scorer = metrics.make_scorer(scorer)

        with tqdm.tqdm(range(len(estimators_list))) as process:
            for i in range(len(estimators_list)):
                start_time = time.time()
                # Not using GridSearch class, but to use cross_validate to get best models.
                estimator = estimators_list[i] 
                
                # Training happen here for each algorithm with n-fold CV!
                cv_result = cross_validate(estimator=estimator, 
                                        X=x, y=y, cv=2, 
                                        scoring=scorer,
                                        return_train_score=True)

                mean_train_score = round(cv_result['train_score'].mean(), 6)
                mean_test_score = round(cv_result['test_score'].mean(), 6)

                # Noted: CV is just to get training info, but real estimator hasn't been fitted!
                # Real training should happen
                estimator.fit(x, y)

                # Now for score_dict key is: {name+train_score: [instance, test_score]}
                estimator_train_name = estimator.name + "_" + str(mean_train_score)
                self.score_list.append((estimator_train_name, estimator, mean_test_score))
                self.estimator_list.append(estimator)

                logger.info("GridSearch for algorithm: {} takes {} seconds".format(estimator_train_name, round(time.time() - start_time, 2)))

                process.update(1)

        # Add with neural network search to find with Neural models
        if self.use_neural_network:
            logger.info("Start to use Nueral Network to fit data with `tuner`!")

            # `fit` related func like validation and save models are warpped in `fit` func, 
            # here just `fit`
            neural_model = NeuralModelSearch(models_path=self.backend.output_folder, task_type=self.task_type) 
            neural_model.fit(x, y)  

            # Add with neural network's score list, so that we could do evaluation for DNN models.
            self.score_list.extend(neural_model.score_list)
            self.estimator_list.extend(neural_model.model_list)
            logger.info("Finished Nueral Network search logic!") 

        logger.info("Model selection training has finished.")

        # after the training finished, then we should get each estimator with score that is based on `training score`
        # and store the score with each instance class name and score.
        # self._get_estimators_score()

        # Here add with information for `n_best_model` name and score
        logger.info("Get some best model scores information based on model_selection module.")
        for alg_name, _, alg_test_score in self.score_list:
            logger.info("Algorithm: {} with test score: {}".format(alg_name, alg_test_score))

        # after we have get the score, then we should store the trained estimators
        logger.info("Start to save best selected models into disk.")
        # remove duplicate ones
        # self.estimator_list = list(set(self.estimator_list))
        self.save_best_model_list()

        return self

    def score(self, x, y):
        """
        To use best fitted model to evaluate test data
        :param x:
        :param y:
        :return:
        """
        scorer = get_scorer_based_on_target(self.task_type)
        
        pred = self.predict(x)

        return scorer(y, pred)

    def predict(self, x):
        """
        Get prediction based on best fitted model
        :param x:
        :return:
        """
        best_estimator = self.best_estimator

        if hasattr(best_estimator, 'predict'):
            return best_estimator.predict(x)
        else:
            raise NotImplementedError("For estimator:{} doesn't support `predict` func!".format(best_estimator))

    def predict_proba(self, x):
        """
        Get probability of based on best estimator
        :param x:
        :return:
        """
        best_estimator = self.best_estimator

        if hasattr(best_estimator, 'predict_proba'):
            return best_estimator.predict_proba(x)
        else:
            raise NotImplementedError("For estimator:{} doesn't support `predict_proba` func!".format(best_estimator))

    def save_best_model_list(self):
        """
        save whole best fitted model based on each algorithm own parameters, so
        that we could save each best models.
        Then we could do ensemble logic.

        Here I think I could just save the each best parameters trained model
        into disk, also the file name should be like `LogisticRegression_9813.pkl`:
        with `classname_score.pkl`.

        Noted: This func should only be called after trained
        :param n_best_model: How many best model to save
        :return:
        """
        # Big change here: Not to get n_best models for each algorithms, but should only get 
        # `n_best` base on test score for full algorithms list!!!
        alg_list_sati = [(alg_name, estimator, test_score) for alg_name, estimator, test_score in self.score_list 
                    if alg_name.split("_")[0] != 'sequential']
        # alg_list_sati.sort(key=lambda l:l[-1], reverse=True)
        # alg_list = alg_list_sati[:self.n_best_model]
        alg_list = sorted(alg_list_sati, key=lambda l: l[-1], reverse=True)[:self.n_best_model]

        if len(alg_list) == 0:
            logger.warning("There isn't any trained model to save except with Nueral Network!")
            return
        if isinstance(alg_list, tuple):
            # in case there is just one tuple
            alg_list = [alg_list]

        # Start looping
        for i, (alg_name, estimator, test_score) in enumerate(alg_list):
            if i == 0:
                print("Get best estimator", estimator)
                print("Get best score:", test_score)
                self.best_estimator = estimator
                self.best_score = test_score

            # alg_name, estimator, test_score = alg_list[i][0], alg_list[i][1], alg_list[i][2]
            alg_name_split = alg_name.split('_')
            alg_name = alg_name_split[0] + "_" + str(test_score)

            logger.info("Start to save model: {}".format(alg_name))
            self.backend.save_model(estimator, alg_name)

        logger.info("Already have saved models: %s" % '\t'.join(self.backend.list_models()))
        
    def load_best_model_list(self, model_extension='pkl'):
        """
        Load previous saved best model into a list of trained instance.
        :return:
        """
        # This should be changed to just load full models from disk, not based on the trained object.
        # As there maybe a failure to save models.
        model_list = self.backend.load_models_combined_with_model_name()

        # To ensure there should be at least one file for `Ensemble` logic.
        if not model_list:
            # In case there isn't any trained model!
            logger.error("There isn't any trained model to load!")
            return None

        model_list = sorted(model_list, key=lambda x: float(x[0].split("_")[1].replace(".pkl", '')), reverse=True)

        return model_list

    def save_bestest_model(self):
        """
        dump best trained model into disk, one more thing here: We shouldn't save
        the model into disk with fixed name, but should with score.
        :return:
        """
        self.best_esmator_name = self.best_estimator.__class__.__name__ + '-' + str(self.best_score) + ".pkl"
        self.backend.save_model(self.best_estimator,  self.best_esmator_name)

    def load_bestest_model(self):
        """
        load best trained model from disk
        :return:
        """
        if self.best_esmator_name is None:
            raise ValueError("When try to load best model from disk, please try to save it frist!")

        return self.backend.load_model(self.best_esmator_name)

    def best_estimator(self):
        return self.best_estimator
   
    def best_score(self):
        return self.best_score

    @staticmethod
    def _score_with_estimator(estimator_instance, x, y):
        """
        Just to get score with trained estimator based on data and label
        :param estimator_instance:
        :param x:
        :param y:
        :return:
        """
        try:
            score = estimator_instance.score(x, y)
            return score
        except Exception as e:
            raise Exception("When try to get score with estimator: {} "
                            "get error: {}".format(estimator_instance.__class__.__name__, e))

    def save_trained_estimator(self, estimator, estimator_name):
        """
        To save the trained model into disk with file name
        :param estimator:
        :param estimator_name:
        :return:
        """
        self.backend.save_model(estimator, estimator_name)

    def print_estimators(self):
        """
        To list whole grid models instances, so that we could check.
        :return:
        """
        logger.info("Get {} estimators.".format(len(self.estimator_list)))
        for grid_estimator in self.estimator_list:
            print(grid_estimator)


if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_digits
    from .classifier_algorithms import LogisticRegression
    from .classifier_algorithms import GradientBoostingTree, LightGBMClassifier
    from utils.backend_obj import Backend

    backend = Backend(output_folder=r"C:\Users\guangqiiang.lu\Downloads\test_automl")

    x, y = load_digits(return_X_y=True)

    g = GridSearchModel(backend)
    lr = LogisticRegression()
    clf = LightGBMClassifier()
    g.add_estimator(lr)
    g.add_estimator(clf)

    g.fit(x, y)
    # print(g.best_estimator)
    # print(g.best_score)
    # print(g.score(x, y))

    # g.save_bestest_model()
    # bst_model = g.load_bestest_model()
    # print(bst_model.score(x, y))
    # print(g.score_dict)
    # g.save_best_model_list()
    # print(g.load_best_model_list())
