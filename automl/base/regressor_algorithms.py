# -*- coding:utf-8 -*-
"""
Let's just use whole regression used in sklearn to be instant
here so that we could change or do something change could be easier.

@author: Guangqiang.lu
"""
import numpy as np
import warnings
from sklearn.base import BaseEstimator
from sklearn import metrics

from hyper_config import (ConfigSpace, UniformHyperparameter, CategoryHyperparameter,
                                                         NormalHyperameter, GridHyperparameter)

warnings.simplefilter('ignore')


class RegressorClass(BaseEstimator):
    def __init__(self):
        """TODO: Change here. Here should be changed, as `self.estimator` will be constructed from `fit`, if
        we have saved trained instance into disk, then re-load won't get the `estimator`!
        """
        super(RegressorClass, self).__init__()
        self.name = self.__class__.__name__
        self.estimator = None
        # Add this for used `Ensemble` logic will check the estimator type.
        self._estimator_type = 'regressor'

    def fit(self, x, y):
        self.estimator.fit(x, y)
        return self

    def predict(self, x):
        try:            
            pred = self.estimator.predict(x)
        except:
            prob = self.predict_proba(x)
            pred = np.argmax(prob, axis=1)

        return pred

    def score(self, x, y):
        scorer = metrics.mean_squared_error

        pred = self.predict(x)
        score = scorer(y, pred)

        return score

    @staticmethod
    def get_search_space():
        """
        This is to get predefined search space for different algorithms,
        and we should use this to do cross validation to get best fitted
        parameters.
        :return:
        """
        raise NotImplementedError


class RegressorFactory:
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
            if alg_name == 'LinearRegressor':
                algorithm_instance_list.append(LinearRegressor())
            elif alg_name == 'SupportVectorRegressor':
                algorithm_instance_list.append(SupportVectorRegressor())
            elif alg_name == 'GBRegressor':
                algorithm_instance_list.append(GBRegressor())
            elif alg_name == 'RFRegressor':
                algorithm_instance_list.append(RFRegressor())
            elif alg_name == 'KNNRegressor':
                algorithm_instance_list.append(KNNRegressor())
            elif alg_name == 'DTRegressor':
                algorithm_instance_list.append(DTRegressor())
            elif alg_name == 'AdaboostRegressor':
                algorithm_instance_list.append(AdaboostRegressor())
            elif alg_name == 'LightGBMRegressor':
                algorithm_instance_list.append(LightGBMRegressor())
            elif alg_name == 'XGBoostRegressor':
                algorithm_instance_list.append(XGBoostRegressor())

        return algorithm_instance_list


class LinearRegressor(RegressorClass):
    def __init__(self, fit_intercept=True):
        super(LinearRegressor).__init__()
        self.fit_intercept = fit_intercept
        self.name= "LinearRegressor"

        from sklearn.linear_model import LinearRegression

        self.estimator = LinearRegression(fit_intercept=self.fit_intercept)        
        
    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        fit_intercept = CategoryHyperparameter(name="fit_intercept", categories=[True, False])
        # grid = GridHyperparameter(name="C", values=[1, 2, 3])

        config.add_hyper([fit_intercept])

        # config.get_hypers()
        return config.get_hypers()


class SupportVectorRegressor(RegressorClass):
    def __init__(self, C=1.,
                 kernel='rbf',
                 random_state=1234
                 ):
        super(SupportVectorRegressor, self).__init__()
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        self.name = "SupportVectorRegressor"

        from sklearn.svm import SVR

        self.estimator = SVR(C=self.C,
                             kernel=self.kernel,
                            )

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        c_list = UniformHyperparameter(name="C", low=0.1, high=100, size=3)
        # kernal = CategoryHyperparameter(name="kernal", categories=["rbf", "linear"])
        # dual = CategoryHyperparameter(name="dual", categories=[True, False])
        # grid = GridHyperparameter(name="C", values=[10, 100])

        config.add_hyper([c_list])

        return config.get_hypers()


class RFRegressor(RegressorClass):
    def __init__(self, n_estimators=100,
                 max_depth=None,
                 max_features='auto',
                 class_weight=None):
        super(RFRegressor, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.class_weight = class_weight
        self.name = "RFRegressor"

        from sklearn.ensemble import RandomForestRegressor

        self.estimator = RandomForestRegressor(n_estimators=self.n_estimators,
                                                max_depth=self.max_depth,
                                                max_features=self.max_features)

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        n_estimator_list = GridHyperparameter(name='n_estimators', values=[100, 300, 500])

        config.add_hyper(n_estimator_list)

        return config.get_hypers()


class GBRegressor(RegressorClass):
    def __init__(self,
                 n_estimators=100,
                 max_depth=3,
                 max_features=None,
                 learning_rate=0.1):
        super(GBRegressor, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.name = "GBRegressor"

        from sklearn.ensemble import GradientBoostingRegressor

        self.estimator = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                                    max_depth=self.max_depth,
                                                    max_features=self.max_features,
                                                    learning_rate=self.learning_rate)

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        n_estimator_list = GridHyperparameter(name='n_estimators', values=[100, 300, 500])
        learning_rate = UniformHyperparameter(name='learning_rate', low=0.01, high=1.0, size=2)

        config.add_hyper([n_estimator_list, learning_rate])

        return config.get_hypers()


class KNNRegressor(RegressorClass):
    def __init__(self, n_neighbors=5):
        super(KNNRegressor).__init__()
        self.n_neighbors = n_neighbors
        self.name = "KNNRegressor"

        from sklearn.neighbors import KNeighborsRegressor

        self.estimator = KNeighborsRegressor(n_neighbors=self.n_neighbors)

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        n_neighbors = GridHyperparameter(name='n_neighbors', values=[3, 5, 7, 10])

        config.add_hyper([n_neighbors])

        return config.get_hypers()


class DTRegressor(RegressorClass):
    def __init__(self, criterion='mse', max_depth=None, max_leaf_nodes=None, min_samples_leaf=1):
        super(DTRegressor).__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.name = "DTRegressor"

        from sklearn.tree import DecisionTreeRegressor

        self.estimator = DecisionTreeRegressor(criterion=self.criterion, 
                                                max_depth=self.max_depth,
                                                max_leaf_nodes=self.max_leaf_nodes,
                                                min_samples_leaf=self.min_samples_leaf)

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        max_depth = GridHyperparameter(name='max_depth', values=[3, 5])

        config.add_hyper([max_depth])

        return config.get_hypers()                                                


class AdaboostRegressor(RegressorClass):
    def __init__(self, base_estimator=None, learning_rate=1.0, n_estimators=50):
        super(AdaboostRegressor).__init__()
        self.base_estimator = base_estimator
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.name = "AdaboostRegressor"

        from sklearn.ensemble import AdaBoostRegressor

        self.estimator = AdaBoostRegressor(base_estimator=self.base_estimator, 
            learning_rate=self.learning_rate, 
            n_estimators=self.n_estimators)
    
    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        # Let's try base estimator with DT, as in real world that also make improvement.
        base_estimators = GridHyperparameter(name='base_estimator', values=[None, DTRegressor()])
        n_estimators = GridHyperparameter(name='n_estimators', values=[50, 70, 100])
        learning_rate = UniformHyperparameter(name='learning_rate', low=0.01, high=1.0, size=2)

        config.add_hyper([n_estimators, learning_rate])

        return config.get_hypers()     

    
class LightGBMRegressor(RegressorClass):
    def __init__(self, n_estimators=100, 
            boosting_type='gbdt', 
            num_leaves=31, 
            max_depth=-1, 
            class_weight=None, 
            learning_rate=.1, 
            n_jobs=-1, 
            random_state=None):
        super(LightGBMRegressor).__init__()
        self.n_estimators = n_estimators
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.random_state = random_state if random_state is not None else np.random.seed(1234)
        self.name = "LightGBMRegressor"

        from lightgbm import LGBMRegressor

        self.estimator = LGBMRegressor(n_estimators=self.n_estimators,
                                            boosting_type=self.boosting_type, 
                                            num_leaves=self.num_leaves,
                                            max_depth=self.max_depth,
                                            class_weight=self.class_weight,
                                            learning_rate=self.learning_rate,
                                            n_jobs=self.n_jobs,
                                            random_state=self.random_state)

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        n_estimators = GridHyperparameter(name='n_estimators', values=[100, 300, 500])
        learning_rate = UniformHyperparameter(name='learning_rate', low=0.01, high=1.0, size=2)

        config.add_hyper([n_estimators, learning_rate])

        return config.get_hypers()


class XGBoostRegressor(RegressorClass):
    def __init__(self, n_estimators=100, 
                    learning_rate=0.1,
                    objective='reg:squarederror',
                    reg_lambda=1, 
                    reg_alpha=0, 
                    gamma=0,
                    eval_metric='mlogloss'):
        super(XGBoostRegressor).__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.objective = objective
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        # Whether or not to see logs, `verbose` is not used for xgboost higher version, set with `eval_metric` to avoid warnings
        self.eval_metric = eval_metric
        self.name = "XGBoostRegressor"

        from xgboost import XGBRegressor

        self.estimator = XGBRegressor(n_estimators=self.n_estimators, 
                                        learning_rate=self.learning_rate, 
                                        reg_lambda=self.reg_lambda,
                                        objective=self.objective,
                                        gamma=self.gamma,
                                        reg_alpha=self.reg_alpha,
                                        eval_metric=self.eval_metric)

    @staticmethod
    def get_search_space():
        config = ConfigSpace()

        n_estimators = GridHyperparameter(name='n_estimators', values=[100, 300, 500])
        learning_rate = UniformHyperparameter(name='learning_rate', low=0.01, high=1.0, size=2)

        config.add_hyper([n_estimators, learning_rate])

        return config.get_hypers()                                 


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import GridSearchCV

    x, y = load_boston(return_X_y=True)

    lr = LinearRegressor()

    print(lr.get_search_space())

    grid = GridSearchCV(lr, param_grid=lr.get_search_space())
    grid.fit(x, y)

    print(grid.score(x, y))
    print(grid.best_estimator_)

    print(hasattr(lr, 'get_search_space'))

    # test with factory pattern.
    alg_name_list = ['LinearRegressor']

    print(RegressorFactory.get_algorithm_instance('LinearRegressor'))
