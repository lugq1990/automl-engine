# -*- coding:utf-8 -*-
"""
To do ensemble logic with whole trained model, try to improve whole score based on
different processing logic, also if we could get better result, then we are lucky!

One more thing, this should be called only and after the pipeline has finished, so that
we could load the trained model from disk, so this should be called from the parent
automl training logic.

@author: Guangqiang.lu
"""
import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import cross_validate
from auto_ml.utils.backend_obj import Backend
from auto_ml.metrics.scorer import accuracy, r2
from auto_ml.base.classifier_algorithms import ClassifierClass, ClassifierFactory
from auto_ml.utils.logger import create_logger
from auto_ml.utils.paths import load_yaml_file


logger = create_logger(__file__)


class ModelEnsemble(ClassifierClass):
    """
    Currently I want to support 2 different ensemble logic:
    Voting(weight combine classification: with soft voting and hard voting,
    regression: weight multiple)
    stacking(add trained model prediction into training data)
    """
    def __init__(self, backend, task_type='classification', ensemble_alg='voting',
                 voting_logic='soft'):
        """
        Based on different task to do different logic.
        :param task_type: which task to do: classification or regression.
        :param ensemble_alg: which ensemble logic to use: `voting` or `stacking`.
        :param voting_logic: whether with `hard` or `soft` voting
        """
        super().__init__()
        self.task_type = task_type
        self.ensemble_alg = ensemble_alg
        self.voting_logic = voting_logic
        # To load and save models
        # self.backend = backend if backend is not None else Backend()
        if backend is None:
            raise ValueError("When to use Model Ensemble class, we get a None `backend` object! Please check!")
        self.backend = backend

        self.model_list = self._load_trained_models()
        # define matrics based on task
        self.metrics = None
        if self.task_type == 'classification':
            self.metrics = accuracy
        elif self.task_type == 'regression':
            self.metrics = r2
        self.estimator = None
        # Whole trained model object list, like: [('LogisticRegression', LogisticRegression-9323.pkl),...]
        self.model_list_without_score = self._get_model_list_without_score()
        # Add attr for `stacking` logic to store the whole models needed to be load
        # for later step to make new dataset, so here just store the instances don't need to re-load
        self.stacking_models = [model[1] for model in self.model_list_without_score] \
            if ensemble_alg == 'stacking' else None

    def fit(self, x, y, **kwargs):
        # First we should to get whole trained models no matter for `voting` or `stacking`
        # self.model_list_without_score = self._get_model_list_without_score()

        if self.ensemble_alg == 'voting':
            self.fit_bagging(x, y, **kwargs)
        elif self.ensemble_alg == 'stacking':
            self.fit_stacking(x, y, **kwargs)

    def fit_bagging(self, x, y, **kwargs):
        """
        Here with ensemble logic like `hard` by number voting or
        `soft` by weight combine.

        For bagging fitting, if we face with classification problem,
        then we could use `voting` logic to get ensemble prediction,
        if regression, then will get weights * each model prediction.
        """
        # Here change logic with voting
        if self.task_type == 'classification':
            if self.voting_logic not in ['hard', 'soft']:
                raise ValueError("For ensemble logic, only `hard` and soft is supported "
                                 "when use `voting` logic.")

            self.estimator = VotingClassifier(estimators=self.model_list_without_score,
                                                voting=self.voting_logic)

            # start to fit the voting estimator
            self.estimator.fit(x, y)

            # Here add logic with saving trained model into disk.
            # TODO: but how to evaluate current model with score? Use cross-valiation to do score
            # get voting model score based on CV result!
            score = cross_validate(self.estimator, x, y, cv=3)['test_score'].mean()
            score_str = str(round(score, 6))
            logger.info("Model ensemble Cross-valiation score: {}".format(score_str))

            store_model_name = 'Voting_{}-{}'.format(self.voting_logic, score_str)

            logger.info("Start to save trained model: {} into disk.".format(store_model_name))
            self.backend.save_model(self.estimator, store_model_name)

        elif self.task_type == 'regression':
            pass

    def fit_stacking(self, x, y, **kwargs):
        """
        Implement with stacking logic is combined trained model prediction and original data into
        a new training data.

        # TODO: how TO choose new algorithm for the training?
        # Just to select the best score algorithm for the later step, based on the factory class
        to get the original class name and to load a new classifier.

        :param x: training data
        :param y: training label
        :param kwargs:
        :return:
        """
        # first should create new dataset.
        x_new = self.create_stacking_dataset(x, backend=self.backend)

        logger.info("Before stacking we have data dimention: {}, "
                    "after stacking we have :{}".format(x.shape[1], x_new.shape[1]))

        # Here we should get best score algorithm name for stacking
        # `model_list_without_score` is sorted based on score in fact.
        # In fact we have to add the estimator name based on what we have(using the yaml file result)
        # as we want to avoid to load the pipeline instance...
        best_estimator_name = self._get_best_model_estimator_name_based_on_yaml()

        logger.info("Get estaimator {} for stacking logic.".format(best_estimator_name))
        # As return is a list, but here we just need `one instance` for combined dataset
        self.estimator = ClassifierFactory.get_algorithm_instance(best_estimator_name)[0]

        logger.info("Loaded {} instance for `stacking` training.".format(best_estimator_name))
        # start training step for `stacking`
        self.estimator.fit(x_new, y)

        # model score should also based on CV result.
        score = cross_validate(self.estimator, x_new, y, cv=5)['test_score'][0]
        score = str(round(score, 6))
        stacking_model_name = "Stacking-{}".format(score)
        logger.info("Stacking model score: {}".format(score))

        self.backend.save_model(self.estimator, stacking_model_name)

    def _get_best_model_estimator_name_based_on_yaml(self):
        """
        Just to get the best estimator name based on the yaml file that we have
        for `stacking` ensemble logic.
        :return:
        """
        algorithm_name_list = load_yaml_file()['classification']['default']
        trained_model_alg_name_list = [x[0].split("-")[0] for x in self.model_list_without_score]

        for algo_name in trained_model_alg_name_list:
            if algo_name in algorithm_name_list:
                return algo_name

        return None

    def _load_trained_models(self):
        """
        To load whole trained model from disk, one more thing, as we also saved
        the processing model into disk, so here should ensure we just load the
        algorithm models.

        Sorted instance object list with name, also we could get the model score
        for later compare use case, this is `sorted` list, so later don't need to
        consider for the order.
        :return:a list of models: [('LR-0.98.pkl', LogisticRegression())]
        """
        model_list = self.backend.load_models_combined_with_model_name()

        # ADD logic: we shouldn't include the `Voting` algorithms instance object in fact
        model_list = [x for x in model_list if not x[0].lower().startswith('voting')]

        # Except Neural network models.
        model_list = [x for x in model_list if not x[0].endswith('.h5')]

        # To ensure there should be at least one file for `Ensemble` logic.
        if not model_list:
            raise IOError("There isn't any trained model for `Ensemble`.")

        # after we have get the model list, we should ensure the model by the model
        # name with endswith score.
        # Model name like this: ('lr_0.98.pkl', lr)
        # model_list.sort(key=lambda x: float("0." + x[0].split('-')[1].split('.')[0]), reverse=True)
        model_list.sort(key=lambda x: float(x[0].split('-')[1].split('.')[0]))

        return model_list

    def get_model_score_list(self):
        """
        To get each model accuracy score list for later compare
        :return:
        """
        score_list = []

        for model_name, _ in self._load_trained_models():
            model_score = model_name.split('.')[0].split('-')[-1]
            score_list.append(model_score)

        return score_list

    def _get_model_list_without_score(self):
        """
        To get the model list without score for ensemble use case.
        :return: [('LogisticRegression', LogisticRegression-9323.pkl),...]
        """
        model_list_without_score = []
        for estimator_tuple in self.model_list:
            estimator_name = estimator_tuple[0].split('-')[0]
            # we shouldn't include `ensemble` models.
            if not estimator_name.lower().startswith('votinig') \
                    and not estimator_name.lower().startswith('stacking'):
                model_list_without_score.append((estimator_name, estimator_tuple[1]))

        return model_list_without_score

    @classmethod
    def create_stacking_dataset(cls, x, backend, task_type='classification', ensemble_alg='stacking'):
        """
        What I want is to create the new dataset based on the whole instances for `stacking`.

        We could use the class func to create this.
        As `stacking` will add new features based on trained models.
        Should make the attr `stacking_models` with the `models instance`.
        :param x:
        :param task_type:
        :param ensemble_alg:
        :return:
        """
        model_ensemble = cls(backend=backend, task_type=task_type, ensemble_alg=ensemble_alg)

        # we don't need to care about `model_list_without_score` has instance or not, as parent does this.
        n_estimators = len(model_ensemble.model_list_without_score)

        # Whole trained estimator prediction result.
        pred_array = np.empty((len(x), n_estimators))

        logger.info("Start to get trained model prediction for `stacking`")
        model_list_without_score = model_ensemble.model_list_without_score
        for i in range(n_estimators):
            logger.info("To get prediction for estimator: {}".format(model_list_without_score[i][0]))

            estimator = model_list_without_score[i][1]

            pred = estimator.predict(x)
            pred_array[:, i] = pred

        # Then should combined the prediction and original data
        x_new = np.concatenate([x, pred_array], axis=1)

        return x_new


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from auto_ml.test.get_test_data import get_training_data

    x, y = load_iris(return_X_y=True)
    x, y = get_training_data()

    model_ensemble = ModelEnsemble(ensemble_alg='stacking', voting_logic='soft', )

    model_ensemble.fit(x, y)
    # print([x[1].__class__ for x in model_ensemble.model_list])
    # print(model_ensemble.model_list)
    # for models in model_ensemble.model_list:
    #     model = models[1]
    #     print(model)
    #     print(getattr(model, "_estimator_type", None))
    print(model_ensemble.stacking_models)

    print(ModelEnsemble.create_stacking_dataset(x))
