# -*- coding:utf-8 -*-
"""
Main Cross-validation class for adding more estimators and get score using CV.

This will contain some model selection logic should be used here like
Grid search logic, as here what I could do is create the search space.
So maybe Grid search to find whole models.

Classifier real training happens here.

@author: Guangqiang.lu
"""
import numpy as np
import time
import tqdm
import itertools
from sklearn.model_selection import GridSearchCV, cross_validate

from auto_ml.utils.backend_obj import Backend
from auto_ml.utils.logger import create_logger


logger = create_logger(__file__)


class GridSearchModel(object):
    """
    Here could just to implement that could add a list
    of estimators and their parameters list.
    I want to make this class to do real training part.
    """
    def __init__(self, backend, n_best_model=None):
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
        self.estimator_list = []
        self.score_list = []
        self._estimator_param_list = []
        self.n_best_model = 10 if n_best_model is None else n_best_model

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
        This is real training logic happens here, also we could
        use parallel training for these estimators.
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

        with tqdm.tqdm(range(len(estimators_list))) as process:
            start_time = time.time()
            for i in range(len(estimators_list)):
                
                # Not using GridSearch class, but to use cross_validate to get best models.
                estimator = estimators_list[i]
                # Training happen here for each algorithm with n-fold CV!
                cv_result = cross_validate(estimator=estimator, 
                                        X=x, y=y, cv=3, 
                                        return_train_score=True, 
                                        return_estimator=True)

                mean_train_score = round(cv_result['train_score'].mean(), 6)
                mean_test_score = round(cv_result['test_score'].mean(), 6)

                # Now for score_dict key is: {name+train_score: [instance, test_score]}
                estimator_train_name = estimator.name + "_" + str(mean_train_score)
                self.score_list.append((estimator_train_name, estimator, mean_test_score))
                self.estimator_list.append(estimator)
                # self.score_dict[estimator.name + str(mean_train_score)] = (estimator, mean_test_score)
                # self.estimator_list = estimator

                # estimator = self.estimator_list[i]
                # estimator.fit(x, y)
                logger.info("GridSearch for algorithm: {} takes {} seconds".format(estimator_train_name, round(time.time() - start_time, 2)))

                process.update(1)

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
        self.save_best_model_list()

        return self

    def score(self, x, y):
        """
        To use best fitted model to evaluate test data
        :param x:
        :param y:
        :return:
        """
        best_estimator = self.best_estimator

        return best_estimator.score(x, y)

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
        # Loop for each algorithm and save `n_best_models` instance. 
        alg_name_set = set([instance_name.split("_")[0] for instance_name, _, _ in self.score_list])
        
        # print(self.score_list)
        for alg in alg_name_set:
            # Get which algorithm, then sort based on test score, get `n_best_models`
            alg_list = [(alg_name, alg_instance, test_score) for alg_name, alg_instance, test_score 
                    in self.score_list if alg_name.startswith(alg)]

            if len(alg_list) == 0:
                raise ValueError("Couldn't get algorithm: {} from `self._score_list`".format(alg))
            
            # sort models based on test score.
            alg_list = sorted(alg_list, key=lambda l: l[-1], reverse=True)
    
            if len(alg_list) > self.n_best_model:
                alg_list = alg_list[self.n_best_model]

            for i in range(len(alg_list)):
                # Loop for satisfied algorithm instance and dump each of them.
                alg_name, estimator, test_score = alg_list[i][0], alg_list[i][1], alg_list[i][2]
                # change trained score with test_score.
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
        # TODO: Why not just load whole models from disk then sort based on name?
        model_list = []

        alg_name_set = set([instance_name.split("_")[0] for instance_name, _, _ in self.score_list])
        
        for alg in alg_name_set:
            # Get which algorithm, then sort based on test score, get `n_best_models`
            alg_list = [(alg_name, alg_instance, test_score) for alg_name, alg_instance, test_score 
                    in self.score_list if alg_name.startswith(alg) ]
            alg_list = sorted(alg_list, key=lambda l: l[-1], reverse=True)[self.n_best_model]

            if len(alg_list) == 0:
                continue
            
            for alg_name, estimator, test_score in alg_list:
                # change trained score with test_score.
                alg_name_split = alg_name.split('_')
                model_name = alg_name_split[0] + "_" + test_score

                logger.info("Start to save model: {}".format(model_name))
                model = self.backend.load_model(model_name)
                model_list.append(model)

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
        return self.backend.load_model(self.best_esmator_name)

    @property
    def best_estimator(self):
        """
        To get best estimator based on the testing score list
        :return: best estimator
        """
        if len(self.score_list) == 0:
            raise ValueError("Please fit model first!")

        max_test_score = np.argmax([test_score for _, _, test_score in self.score_list])

        try:
            return self.estimator_list[max_test_score]
        except IndexError as e:
            logger.error("To get best estimator with index error: {}".format(e))
            raise IndexError("To get best estimator with index error: {}".format(e))

    @property
    def best_score(self):
        if len(self.score_list) == 0:
            raise ValueError("Please fit model first!")
        
        try:
            max_score = max([test_score for _, _, test_score in self.score_list])
            return max_score    
        except IndexError as e:
            logger.error("To get best score with index error: {}".format(e))
            raise IndexError("To get best score with index error: {}".format(e))
        
    # def _get_estimators_score(self):
    #     """
    #     To get whole trained estimator based on data and label for storing
    #     the result based on each trained grid model best estimator.
    #     score_dict is like: {'LogisticRegressin': (lr, 0.9877)}
    #     :return:
    #     """
    #     for estimator in self.estimator_list:
    #         # here I also need the trained estimator object, so here
    #         # also with trained object.
    #         best_estimator = estimator.best_estimator_
    #         best_score = round(estimator.best_score_, 6)

    #         # This should based on the trained best score, not based on trained model then score again.
    #         self.score_dict[best_estimator.__class__.__name__] = (best_estimator, best_score)
    #                                            #self._score_with_estimator(best_estimator, x, y))

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
    from sklearn.datasets import load_iris
    from auto_ml.base.classifier_algorithms import LogisticRegression
    from auto_ml.base.classifier_algorithms import GradientBoostingTree
    from auto_ml.utils.backend_obj import Backend

    backend = Backend()

    x, y = load_iris(return_X_y=True)

    g = GridSearchModel(backend)
    lr = LogisticRegression()
    clf = GradientBoostingTree()
    g.add_estimator(lr)
    # g.add_estimator(clf)

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
