# -*- coding:utf-8 -*-
"""
This is the whole score functionality that could be used for metrics.

author: Guangqiang.lu
"""

from abc import ABCMeta

import numpy as np
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target


class Scorer(object, metaclass=ABCMeta):
    """
    This class is used to make the scorer function to instance
    scorer object
    """
    def __init__(self, name, score_func, max_or_min, sign, kwargs):
        """
        init scorer object
        :param name: which name the metric is
        :param score_func: which scorer function to use
        :param max_or_min: as for different problem, as for classification
        problem best score is 1, and for regression problem min score is 0
        :param sign: the sign object for metrics, like -1 or 1
        :param kwargs: other key words
        """
        self.name = name
        self._score_func = score_func
        self._max_or_min = max_or_min
        self._sign = sign
        self._kwargs = kwargs

    @classmethod
    def __call__(self, y_true, y_pred, sample_weights=None, *args, **kwargs):
        """
        This is the abstract function that subclass should implement
        :param y_true: True label array
        :param y_pred: prediction label array
        :param sample_weights: each sample weights, default is None
        :param args: other keys
        :param kwargs: other key words
        :return: score computed with self._score_func
        """
        pass

    def __str__(self):
        return self.name


# Here is to make the metric class for supporting prediction evaluation,
# probability evaluation and threshould evaluation
class _PredScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
        """
        implement the Scorer caller function, just support for classification,
        not support for regression
        :param y_true: array-like, [n_sampels,] or [n_samples * n_classes]
            true label array
        :param y_pred: array-like, [n_samples,] or [n_samples * n_classes]
            prediction label array
        :param sample_weight: array-like, optional, default is None
            each sample weights, default is None
        :param args:
            other keys
        :param kwargs:
            other key words
        :return:
            computed score with prediction
        """
        true_type = type_of_target(y_true)
        if len(y_true.shape) == 1 and true_type == 'continous':
            # not used for regression problem
            pass
        elif true_type == 'binary' or true_type == 'multiclass':
            if len(y_pred.shape) != 1:
                # if this is one-hot type object
                y_pred = np.argmax(y_pred, axis=1)
        elif true_type == 'multilabel-indicator':
            # here is multi-label classification
            y_pred[y_pred > .5] = 1.
            y_pred[y_pred <= .5] = 0.

        if sample_weight is not None:
            # if we could multiply each sample prediction with weights
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 **kwargs)


class _ProbScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
        """
        this class is used to compute the score with probability based,
        y_pred should be the probability!
        :param y_true: array-like, [n_samples * n_classes]
            true label array
        :param y_pred: array-like, [n_samples * n_classes]
            probability array
        :param sample_weight: array-like, optional, default is None
            weights for each sample
        :param args:
            other keys
        :param kwargs:
            other key words
        :return:
            computed score with probability
        """
        true_type = type_of_target(y_true)
        if len(y_true.shape) == 0 or true_type == 'continous':
            pass

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 **kwargs)


class _ThreaScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
        """
        this class is used to compute the score with probability based,
        y_pred should be the probability!
        :param y_true: array-like, [n_samples * n_classes]
            true label array
        :param y_pred: array-like, [n_samples * n_classes]
            probability array
        :param sample_weight: array-like, optional, default is None
            weights for each sample
        :param args:
            other keys
        :param kwargs:
            other key words
        :return:
            computed score with probability
        """
        y_type = type_of_target(y_true)
        if y_type not in ('binary', 'multilabel-indicator'):
            raise ValueError("{} format is not supported".format(y_type))

        if y_type == 'binary':
            # we chould just get the last column probability
            y_pred = y_pred[:, 1]
        elif y_type == 'multilabel-indicator':
            y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 **kwargs)


# here is a helper function to make scorer for later step use case
def make_scorer(name, score_func, max_or_min=1, greater_is_better=True,
                need_proba=False, need_thre=False, **kwargs):
    """
    This is to make scorer object just like sklearn make_scorer
    :param name: string-type
        the name for the function
    :param score_func: callable function object
        as sklearn score object, `score_func(y_true, y_pred, **kwargs)`
    :param max_or_min: int or float
        the max score for accuracy and min score for loss function, that means
        if score function is accuracy, then this should be 1, if score function
        is loss function, then this should be 0
    :param greater_is_better: Boolean, True or False
        for different metrics, whether bigger is better, like for accuracy like
        metrics, bigger is better, but for loss function, smaller is better
    :param need_proba: Boolean
        whether or not to use the probability, sometimes like AUC score is based on
        probability, then we should give the proba that metrics need
    :param need_thre: Boolean
        whether or not to use threshold to compute the score like AUC score
    :param kwargs:
        other key words
    :return:
        metrics object
    """
    sign = 1 if greater_is_better else -1
    if need_proba:
        cls = _ProbScorer
    if need_thre:
        cls = _ThreaScorer
    else:
        cls = _PredScorer
    return cls(name, score_func, max_or_min, sign, kwargs)


# Here should create with standard regression metrics
r2 = make_scorer('r2',
                 metrics.r2_score)

mean_absolute_error = make_scorer('mean_absolute_error',
                                  metrics.mean_absolute_error,
                                  max_or_min=0,
                                  greater_is_better=False)

mean_squared_error = make_scorer('mean_squared_error',
                                 metrics.mean_squared_error,
                                 max_or_min=0,
                                 greater_is_better=False)


# here is standard classification problem metrics
accuracy = make_scorer('accuracy', metrics.accuracy_score)

f1 = make_scorer('f1', metrics.f1_score)

roc_auc = make_scorer('roc_auc', metrics.roc_auc_score, need_proba=True)

precision = make_scorer('precision', metrics.precision_score)

recall = make_scorer('recall', metrics.recall_score)

average_precision = make_scorer('average_precision', metrics.average_precision_score)

