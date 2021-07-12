# -*- coding:utf-8 -*-
"""
Let's just use whole classifier used in sklearn to be instant
here so that we could change or do something change could be easier.

@author: Guangqiang.lu
"""
import numpy as np
import warnings
from sklearn.base import BaseEstimator

from auto_ml.hyper_config import (ConfigSpace, UniformHyperparameter, CategoryHyperparameter,
                                                         NormalHyperameter, GridHyperparameter)

warnings.simplefilter('ignore')


class ClassifierFactory:
    """
    Factory design pattern.

    This is used to init with different algorithm name with the instance.
    """
    def __init__(self):
        self.algorithm_instance_list = []

    @staticmethod
    def get_algorithm_instance(alg_name_list):
        """
        To get whole classifier instance list based on the needed algorithms name.
        :param alg_name_list:
        :return:
        """
        algorithm_instance_list = []

        if isinstance(alg_name_list, str):
            # Should be list, but string will be fine
            alg_name_list = [alg_name_list]

        for alg_name in alg_name_list:
            if alg_name == 'LogisticRegression':
                algorithm_instance_list.append(LogisticRegression())
            elif alg_name == 'SupportVectorMachine':
                algorithm_instance_list.append(SupportVectorMachine())
            elif alg_name == 'GradientBoostingTree':
                algorithm_instance_list.append(GradientBoostingTree())
            elif alg_name == 'RandomForestClassifier':
                algorithm_instance_list.append(RandomForestClassifier())
            elif alg_name == 'KNNClassifier':
                algorithm_instance_list.append(KNNClassifier())
            elif alg_name == 'DecisionTreeClassifier':
                algorithm_instance_list.append(DecisionTreeClassifier())
            elif alg_name == 'AdaboostClassifier':
                algorithm_instance_list.append(AdaboostClassifier())
            elif alg_name == 'LightGBMClassifier':
                algorithm_instance_list.append(LightGBMClassifier())
            elif alg_name == 'XGBClassifier':
                algorithm_instance_list.append(XGBClassifier())

        return algorithm_instance_list


class ClassifierClass(BaseEstimator):
    def __init__(self):
        """TODO: Change here. Here should be changed, as `self.estimator` will be constructed from `fit`, if
        we have saved trained instance into disk, then re-load won't get the `estimator`!
        """
        super(ClassifierClass, self).__init__()
        self.name = self.__class__.__name__
        self.estimator = None
        # Add this for used `Ensemble` logic will check the estimator type.
        self._estimator_type = 'classifier'

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        try:            
            pred = self.estimator.predict(x)
        except:
            prob = self.predict_proba(x)
            pred = np.argmax(prob, axis=1)

        return pred

    def score(self, x, y):
        score = self.estimator.score(x, y)
        return score

    def predict_proba(self, x):
        prob = self.estimator.predict_proba(x)
        return prob

    @staticmethod
    def get_search_space():
        """
        This is to get predefined search space for different algorithms,
        and we should use this to do cross validation to get best fitted
        parameters.
        :return:
        """
        raise NotImplementedError


class LogisticRegression(ClassifierClass):
    def __init__(self, C=1.,
                 class_weight=None,
                 dual=False,
                 fit_intercept=True,
                 penalty='l2',
                 n_jobs=None,
                 random_state=1234
                 ):
        super().__init__()
        self.C = C
        self.class_weight = class_weight
        self.dual = dual
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.n_jobs = n_jobs
        self.random_state = random_state

        from sklearn.linear_model import LogisticRegression

        self.estimator = LogisticRegression(C=self.C,
                                class_weight=self.class_weight,
                                dual=self.dual,
                                fit_intercept=self.fit_intercept,
                                penalty=self.penalty,
                                n_jobs=self.n_jobs,
                                random_state=self.random_state)


    def fit(self, x, y):
        
        self.estimator.fit(x, y)
        return self

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        c_list = UniformHyperparameter(name="C", low=0.1, high=100, size=3)
        # dual = CategoryHyperparameter(name="dual", categories=[True, False])
        # grid = GridHyperparameter(name="C", values=[1, 2, 3])

        config.add_hyper([c_list])

        # config.get_hypers()
        return config.get_hypers()


class SupportVectorMachine(ClassifierClass):
    def __init__(self, C=1.,
                 class_weight=None,
                 kernel='rbf',
                 probability=True,
                 random_state=1234
                 ):
        super(SupportVectorMachine, self).__init__()
        self.C = C
        self.class_weight = class_weight
        self.kernel = kernel
        self.probability = probability
        self.random_state = random_state

    def fit(self, x, y):
        from sklearn.svm import SVC

        self.estimator = SVC(C=self.C,
                             class_weight=self.class_weight,
                             kernel=self.kernel,
                             probability=self.probability,
                             random_state=self.random_state
                            )

        self.estimator.fit(x, y)
        return self

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        c_list = UniformHyperparameter(name="C", low=0.1, high=10, size=3)
        # dual = CategoryHyperparameter(name="dual", categories=[True, False])
        # grid = GridHyperparameter(name="C", values=[10, 100])

        config.add_hyper([c_list])

        return config.get_hypers()


class RandomForestClassifier(ClassifierClass):
    def __init__(self, n_estimators=100,
                 max_depth=None,
                 max_features='auto',
                 class_weight=None):
        super(RandomForestClassifier, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.class_weight = class_weight

    def fit(self, x, y):
        from sklearn.ensemble import RandomForestClassifier

        self.estimator = RandomForestClassifier(n_estimators=self.n_estimators,
                                                max_depth=self.max_depth,
                                                max_features=self.max_features,
                                                class_weight=self.class_weight)
        self.estimator.fit(x, y)
        return self

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        n_estimator_list = GridHyperparameter(name='n_estimators', values=[100, 300, 500])

        config.add_hyper(n_estimator_list)

        return config.get_hypers()


class GradientBoostingTree(ClassifierClass):
    def __init__(self,
                 n_estimators=100,
                 max_depth=3,
                 max_features=None,
                 learning_rate=0.1):
        super(GradientBoostingTree, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.learning_rate = learning_rate

    def fit(self, x, y):
        from sklearn.ensemble import GradientBoostingClassifier

        self.estimator = GradientBoostingClassifier(n_estimators=self.n_estimators,
                                                    max_depth=self.max_depth,
                                                    max_features=self.max_features,
                                                    learning_rate=self.learning_rate)
        self.estimator.fit(x, y)
        return self

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        n_estimator_list = GridHyperparameter(name='n_estimators', values=[100, 300, 500])
        learning_rate = UniformHyperparameter(name='learning_rate', low=0.01, high=1.0, size=2)

        config.add_hyper([n_estimator_list, learning_rate])

        return config.get_hypers()


class KNNClassifier(ClassifierClass):
    def __init__(self, n_neighbors=5):
        super().__init__()
        self.n_neighbors = n_neighbors

    def fit(self, x, y, **kwargs):
        from sklearn.neighbors import KNeighborsClassifier

        self.estimator = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        self.estimator.fit(x, y)
        return self

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        n_neighbors = GridHyperparameter(name='n_neighbors', values=[3, 5, 7, 10])

        config.add_hyper([n_neighbors])

        return config.get_hypers()


class DecisionTreeClassifier(ClassifierClass):
    def __init__(self, criterion='gini', max_depth=None, max_leaf_nodes=None, min_samples_leaf=1):
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf

    def fit(self, x, y, **kwargs):
        from sklearn.tree import DecisionTreeClassifier

        self.estimator = DecisionTreeClassifier(criterion=self.criterion, 
                                                max_depth=self.max_depth,
                                                max_leaf_nodes=self.max_leaf_nodes,
                                                min_samples_leaf=self.min_samples_leaf)
        self.estimator.fit(x, y, **kwargs)

        return self

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        max_depth = GridHyperparameter(name='max_depth', values=[3, 5, 10, None])

        config.add_hyper([max_depth])

        return config.get_hypers()                                                


class AdaboostClassifier(ClassifierClass):
    def __init__(self, base_estimator=None, learning_rate=1.0, n_estimators=50):
        super().__init__()
        self.base_estimator = base_estimator
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

    def fit(self, x, y, **kwargs):
        from sklearn.ensemble import AdaBoostClassifier

        self.estimator = AdaBoostClassifier(base_estimator=self.base_estimator, learning_rate=self.learning_rate, n_estimators=self.n_estimators)
        
        self.estimator.fit(x, y, **kwargs)
        return self
    
    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        # Let's try base estimator with DT, as in real world that also make improvement.
        base_estimators = GridHyperparameter(name='base_estimator', values=[None, DecisionTreeClassifier()])
        n_estimators = GridHyperparameter(name='n_estimators', values=[50, 70, 100])
        learning_rate = UniformHyperparameter(name='learning_rate', low=0.01, high=1.0, size=2)

        config.add_hyper([n_estimators, learning_rate])

        return config.get_hypers()     

    
class LightGBMClassifier(ClassifierClass):
    def __init__(self, n_estimators=100, 
            boosting_type='gbdt', 
            num_leaves=31, 
            max_depth=-1, 
            class_weight=None, 
            learning_rate=.1, 
            n_jobs=-1, 
            random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.random_state = random_state if random_state is not None else np.random.seed(1234)
    
    def fit(self, x, y, **kwargs):
        from lightgbm import LGBMClassifier

        self.estimator = LGBMClassifier(n_estimators=self.n_estimators,
                                            boosting_type=self.boosting_type, 
                                            num_leaves=self.num_leaves,
                                            max_depth=self.max_depth,
                                            class_weight=self.class_weight,
                                            learning_rate=self.learning_rate,
                                            n_jobs=self.n_jobs,
                                            random_state=self.random_state)

        self.estimator.fit(x, y, **kwargs)

        return self

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        n_estimators = GridHyperparameter(name='n_estimators', values=[100, 300, 500])
        learning_rate = UniformHyperparameter(name='learning_rate', low=0.01, high=1.0, size=2)

        config.add_hyper([n_estimators, learning_rate])

        return config.get_hypers()


class XGBClassifier(ClassifierClass):
    def __init__(self, n_estimators=100, 
                    learning_rate=0.1,
                    objective='binary:logistic',
                    reg_lambda=1, 
                    reg_alpha=0, 
                    gamma=0,
                    eval_metric='mlogloss'):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.objective = objective
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        # Whether or not to see logs, `verbose` is not used for xgboost higher version, set with `eval_metric` to avoid warnings
        self.eval_metric = eval_metric
    
    def fit(self, x, y, **kwargs):
        from xgboost import XGBClassifier

        self.estimator = XGBClassifier(n_estimators=self.n_estimators, 
                                        learning_rate=self.learning_rate, 
                                        reg_lambda=self.reg_lambda,
                                        objective=self.objective,
                                        gamma=self.gamma,
                                        reg_alpha=self.reg_alpha,
                                        eval_metric=self.eval_metric)

        self.estimator.fit(x, y, **kwargs)
        return self

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        n_estimators = GridHyperparameter(name='n_estimators', values=[100, 300, 500])
        learning_rate = UniformHyperparameter(name='learning_rate', low=0.01, high=1.0, size=2)

        config.add_hyper([n_estimators, learning_rate])

        return config.get_hypers()                                 


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import GridSearchCV

    x, y = load_iris(return_X_y=True)

    lr = LogisticRegression()

    print(lr.get_search_space())

    grid = GridSearchCV(lr, param_grid=lr.get_search_space())
    grid.fit(x, y)

    print(grid.score(x, y))
    print(grid.best_estimator_)

    print(hasattr(lr, 'get_search_space'))

    print(lr.name)

    # test with factory pattern.
    alg_name_list = ['LogisticRegression']

    print(ClassifierFactory.get_algorithm_instance('LogisticRegression'))
