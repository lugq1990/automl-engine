# -*- coding:utf-8 -*-
"""
Whole utils that could be used for data convention transformation.

author: Guangqiang.lu
"""
import numpy as np
import hashlib
import pandas as pd
import scipy.sparse as sp

from sklearn import metrics
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import type_of_target

from .CONSTANT import *
from metrics.scorer import *


def ensure_data_without_nan(data):
    """
    this is used to convert data to ensure that
    there isn't anything nan or inf value.
    replace with inf and nan value with mean value for continuous
    data, with most frequent value for categorical feature.
    with each column.
    :param data: array type
    :return: array data
    """
    # this is to use DataFrame to make do the missing value filling
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    # convert inf to nan value
    data[~np.isfinite(data)] = np.nan

    df = pd.DataFrame(data, columns=range(data.shape[1]))
    missing_cols = list(df.isnull().sum().index)
    df[missing_cols].fillna(df[missing_cols].mean())

    return df.values


def nan_col_index(data):
    """
    this is to get the nan columns index that contain inf value
    :param data: array type
    :return: index with True if nan or inf value
    """
    try:     
        nan_index = np.any(~np.isfinite(data), axis=0)
    except:
        nan_index = np.array([True])
    return nan_index


def is_sparse(data):
    """
    this is to check whether the data is sparse or not.
    :param data: array type
    :return: True is sparse data else False
    """
    if sp.issparse(data):
        return True
    else:
        return False


def check_data_and_label(data, label):
    """
    this is to check with data and label, we should
    ensure the data and label that we need.
    Also we support with sparse data.
    :param data: array-like
    :param label: array-like
    :return: checked data and label
    """
    check_label(label)
    # Here is to ensure that the data is 2D
    data = ensure_2d_data(data)

    # For `check_X_y` is only used for the numerical data type. So data validation shouldn't happen here.
    # Should after the processing logic finished.
    # data, label = check_X_y(data, label, accept_sparse=True)

    return data, label


def ensure_2d_data(data):
    """
    this is to ensure without nan or inf value,
    as for training step, we don't support nan value,
    this function should be used check data step later.
    :param data: array-like
    :return: array or raise error
    """
    if len(data.shape) == 1:
        # we have to ensure data is 2D
        data = data.reshape(-1, 1)

    # for `check_array` is only used for `float` type. Here we just want to ensure the data is 2D
    # return check_array(data)
    return data


def check_label(y):
    """
    this is to ensure we have satisfied target label,
    as we only support with CONSTANT type of target.
    :param y: array like
    :return:
        Type of the problem that are supported with a number based on the CONSTANT module.
        raise error not supported
    """
    # we don't support with target value with nan value
    if nan_col_index(y):
        raise ValueError("We don't support with nan-label! Please check it!")

    type_y = type_of_target(y)
    
    # Change to support current project with `continuous` as regression.
    if type_y == 'continuous':
        type_y = 'regression'

    # current support target type
    supported_type = TASK_TO_STRING.values()
    if type_y not in supported_type:
        raise NotImplementedError("Don't support: %s type of problem" % type_y)

    type_y = STRING_TO_TASK.get(type_y)

    return type_y


def hash_dataset_name(data):
    """
    this is to hash the data to get the name of dataset
    :param data: array-like
    :return: string of a dataset as a name
    """
    md5 = hashlib.md5()
    if sp.issparse(data):
        md5.update(data.data)
        md5.update(str(data.shape).encode('utf-8'))
    else:
        md5.update(data)
    dataset_name = md5.hexdigest()

    return dataset_name


def convert_null_to_none(value):
    """
    this is to convert '' or nan to None for a list
    :param value: list or value
    :return: value
    """
    if isinstance(value, list):
        value = [None if x == '' else x for x in value]
    else:
        value = None

    return value


def is_categorical_type(series):
    """
    To check the data type of the series
    :param series:
    :return:
    """
    # after convert dataframe into array, if there is anything string, then others will be string too.
    # so we couldn't fit with array data type that has string type....
    from pandas.api.types import (is_bool_dtype, is_categorical_dtype, is_object_dtype, is_numeric_dtype,
                                  is_string_dtype, is_datetime64_dtype, is_timedelta64_dtype, is_integer_dtype)

    return is_string_dtype(series) or is_categorical_dtype(series) or \
           is_object_dtype(series) or is_bool_dtype(series) or is_integer_dtype(series)


def get_type_problem(y):
    """
    To check what type of the label dataset.
    :param y:
    :return:
    """
    label_type = check_label(y)

    if label_type in CLASSIFICTION_TASK:
        return 'classification'
    elif label_type in REGRESSION_TASK:
        return 'regression'
    else:
        raise ValueError("When to check label type based label data, "
                         "get not supported type: {}".format(label_type))


def get_scorer_based_on_target(task_type='classification'):
    """
    Based on the target to return the type of the problem.
    :param y:
    :return:
    """
    # for different problem use different scorer
    # self.type_of_problem = self._get_type_problem(y)
    if task_type == 'classification':
        scorer = metrics.accuracy_score
    elif task_type == 'regression':
        # Change to r2 score
        scorer = metrics.r2_score
    else:
        raise ValueError("When to score data get not "
                         "supported type of problem: {}".format(task_type))

    return scorer


def get_num_classes_based_on_label(label):
    if len(label.shape) > 2:
        if label.shape[1] == 1:
            return len(np.unique(label[:, 0]))
        else:
            # currently is with multi-classes only, for multi-label should change this 
            return label.shape[1]
    else:
        return len(np.unique(label))


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)
    from test.get_test_data import get_training_data
    x, y = get_training_data()

    print(hash_dataset_name(x))
    # print(ensure_no_nan_value(x))
    # print(check_data_and_label(x, y))
    print(get_num_classes_based_on_label(y))