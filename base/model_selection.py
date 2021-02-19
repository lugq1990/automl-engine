# -*- coding:utf-8 -*-
"""
This will contain some model selection logic should be used here like
Grid search logic, as here what I could do is create the search space.
So maybe Grid search to find whole models.
Here is the classifier real training happens here.

@author: Guangqiang.lu
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from auto_ml.utils.backend_obj import Backend
from auto_ml.utils.logger import create_logger


logger = create_logger(__file__)


class GridSearchModel(object):
    """
    Here could just to implement that could add a list
    of estimators and their parameters list.
    I want to make this class to do real training part.
    """
    def __init__(self):
        """
        self.estimator_list is like: [GridSearchCV(lr, params), ...]
        self.score_dict is like: {'LogisticRegressin': (lr, 0.9877)}
        """
        super(GridSearchModel, self).__init__()
        # as we need to do training, so here will just store the trained best model
        # for later step ensemble
        self.backend = Backend()
        self.estimator_list = []
        self.score_dict = {}
        self.n_best_model = 10

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
            if self.estimator_list is None:
                self.estimator_list = [GridSearchCV(estimator, estimator.get_search_space())]
            else:
                self.estimator_list.append(GridSearchCV(estimator, estimator.get_search_space()))
        else:
            if estimator_params is None:
                raise ValueError("When we need to set other sklearn native estimator, do"
                                 "need to add estimator_params for searching")

            if self.estimator is None:
                self.estimator_list = [GridSearchCV(estimator, estimator_params)]
            else:
                self.estimator_list.append(GridSearchCV(estimator, estimator_params))

        return self

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

    def fit(self, x, y, n_jobs=None):
        """
        This is real training logic happens here, also we could
        use parallel training for these estimators.
        :param x: training data
        :param y: training label
        :param n_jobs: how much cores to use
        :return:
        """
        if n_jobs is None:
            # as we don't need to use full cores, so here will just
            # to do training sequence for each training estimator
            logger.info("Start to train model based on whole cores.")
            for estimator in self.estimator_list:
                estimator.fit(x, y)
        elif n_jobs:
            # here couldn't use multiprocessing here, just to set
            # estimator `n_job`
            # we could add other multiprocessing here either if we want.
            logger.info("Start to train model based on {} cores.".format(n_jobs))
            for estimator in self.estimator_list:
                if hasattr(estimator, 'n_jobs'):
                    estimator.n_jobs = n_jobs
                estimator.fit(x, y)
        logger.info("Model selection training has finished.")

        # after the training finished, then we should get each estimator with score
        # and store the score with each instance class name and score.
        self._get_estimators_score()

        # Here add with information for `n_best_model` name and score
        logger.info("Get some best model scores information based on model_selection module.")
        for alg_name, score_tuple in self.score_dict.items():
            logger.info("Algorithm: {} with score: {}".format(alg_name, score_tuple[1]))

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
        # added how many best model to be saved into disk based on score.
        n_best_score_dict = {k: v for k, v in sorted(self.score_dict.items(),
                                                     key=lambda x: x[1][1], reverse=True)[:self.n_best_model]}

        for estimator_name, estimator_tuple in n_best_score_dict.items():
            estimator = estimator_tuple[0]
            # NO NEED: just save the score float into str
            # Get score string with round 6 or 100% accuracy
            # score_str = str(round(estimator_tuple[1], 6)).split('.')[-1] \
            #     if estimator_tuple[1] < 1.0 else '100'
            score_str = str(round(estimator_tuple[1], 6))

            # ADD with `-` with name and score, so that we could get the score later
            model_name = estimator_name + "-" + score_str
            logger.info("Start to save model: {}".format(model_name))
            self.backend.save_model(estimator, model_name)

        logger.info("Already have saved models: %s" % '\t'.join(self.backend.list_models()))

    def load_best_model_list(self):
        """
        Load previous saved best model into a list of trained instance.
        :return:
        """
        model_list = []
        n_best_score_dict = {k: v for k, v in sorted(self.score_dict.items(),
                                                     key=lambda x: x[1][1], reverse=True)[:self.n_best_model]}

        for estimator_name, estimator_tuple in n_best_score_dict.items():
            model_name = estimator_name + "-" + str(round(estimator_tuple[1], 6))
            try:
                model_instance = self.backend.load_model(model_name)
                model_list.append(model_instance)
            except IOError as e:
                raise IOError("When try load trained model file:{} "
                              "with error: {}".format(estimator_name, e))
        return model_list

    def save_bestest_model(self):
        """
        dump best trained model into disk, one more thing here: We shouldn't save
        the model into disk with fixed name, but should with score.
        :return:
        """
        self.best_esmator_name = self.best_estimator.__class__.__name__ + '-' + self.best_score
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
        To get best estimator based on the training score list
        :return: best estimator
        """
        best_estimator_list = [estimator.best_estimator_ for estimator in self.estimator_list]
        best_score_list = [estimator.best_score_ for estimator in self.estimator_list]

        if best_estimator_list is not None:
            if best_score_list is None:
                logger.error("We to get best estimator, don't get best score list")
                raise Exception("We to get best estimator, don't get best score list")

            best_estimator = best_estimator_list[np.argmax(best_score_list)]

            return best_estimator
        else:
            logger.error("When to get best estimator, we don't get the best estimator list")
            raise Exception("When to get best estimator, we don't get the best estimator list")

    @property
    def best_score(self):
        best_score_list = [estimator.best_score_ for estimator in self.estimator_list]
        try:
            return round(max(best_score_list), 6)
        except Exception as e:
            raise Exception("When try to get best score get error: {}".format(e))

    def _get_estimators_score(self):
        """
        To get whole trained estimator based on data and label for storing
        the result based on each trained grid model best estimator.
        score_dict is like: {'LogisticRegressin': (lr, 0.9877)}
        :return:
        """
        for estimator in self.estimator_list:
            # here I also need the trained estimator object, so here
            # also with trained object.
            best_estimator = estimator.best_estimator_
            best_score = round(estimator.best_score_, 6)

            # This should based on the trained best score, not based on trained model then score again.
            self.score_dict[best_estimator.__class__.__name__] = (best_estimator, best_score)
                                               #self._score_with_estimator(best_estimator, x, y))

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
    from auto_ml.base.classifier_algorithms import SupportVectorMachine

    x, y = load_iris(return_X_y=True)

    g = GridSearchModel()
    lr = LogisticRegression()
    clf = SupportVectorMachine()
    g.add_estimator(lr)
    g.add_estimator(clf)

    g.fit(x, y)
    print(g.best_estimator)
    print(g.best_score)
    print(g.score(x, y))

    g.save_bestest_model()
    bst_model = g.load_bestest_model()
    print(bst_model.score(x, y))
    print(g.score_dict)
    g.save_best_model_list()
    print(g.load_best_model_list())
