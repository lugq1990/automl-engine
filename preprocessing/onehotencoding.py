# -*- coding:utf-8 -*-
"""
This is used to process with categorical data type to convert it into onehot type

@author: Guangqiang.lu
"""
from pandas.api.types import (is_bool_dtype, is_categorical_dtype, is_object_dtype, is_numeric_dtype,
                              is_string_dtype, is_datetime64_dtype,is_timedelta64_dtype, is_integer_dtype)
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from auto_ml.preprocessing.processing_base import Process


class OnehotEncoding(Process):
    def __init__(self, keep_origin_feature=True, except_feature_indexes=None, except_feature_names_list=None):
        """
        In case there will be numpy array or pandas DataFrame data type,
        so that we could try to use both of them types.
        :param keep_origin_feature: when to transform, to keep original feature or not.
        :param except_feature_indexes: array column indexes
        :param except_feature_names_list: some features doesn't need to convert even they could.
        """
        super(OnehotEncoding, self).__init__()
        self.keep_origin_feature = keep_origin_feature
        self.except_feature_indexes = except_feature_indexes
        self.except_feature_names_list = except_feature_names_list

    def fit(self, x, lable=None):
        """
        To fit the onehot model.
        :param x: data should be DataFrame only! As if we have array that contains string,
            then we couldn't get the type of each column.
        :return:
        """
        # first let me check that data type is dataframe or not
        # if even with dataframe, we still could use `except_feature_indexes` to get these features
        # need to store the data type, as if we fit data is dataframe, if transform is array type, couldn't make it.
        self.data_type = isinstance(x, pd.DataFrame)
        self.data_col_number = x.shape[1]

        if not self.data_type:
            # if this is not a dataframe, but still with  `except_feature_names_list`, we don't handle this.
            if self.except_feature_names_list:
                raise ValueError("With array type, shouldn't with `except_feature_names_list` set!")

        # here we still need to get the categorical features indexes.
        # we have to store the feature_list for later transform use case.
        self.feature_list = self._get_category_features_indexes(x)

        # As even with dataframe, we will just return array type, so let's just convert feature index into indexes.
        if self.data_type:
            feature_name_list = list(x.columns)
            self.feature_list = [feature_name_list.index(x) for x in self.feature_list]

        # Here I can't use pandas, as if we need to do same logic with test data, we need to store the model.
        # return pd.get_dummies(x, prefix=feature_list)

        self.estimator = OneHotEncoder()
        if self.feature_list:
            if self.data_type:
                x = x.iloc[:, self.feature_list]
                # x = x[self.feature_list]
            else:
                x = x[:, self.feature_list]
                if len(self.feature_list) == 1:
                    x = x.reshape(-1, 1)

        self.estimator.fit(x)
        return self

    def transform(self, data, lable=None):
        """
        Should try to keep the data type with same type like array or pandas,
        so even if we fit with pandas, but we still could use array data type to do transform.
        Originally I want to keep the data type also into dataframe if provided dataframe,
        but that's useless, so there just to make it into a array.
        :param data: data could be DataFrame or array type could be fine.
        :return: array type
        """
        # at least we should ensure with same column numbers
        if data.shape[1] != self.data_col_number:
            raise ValueError("When to do transform logic, should provide same dimension, "
                             "fitted data is: {}, current is {}".format(self.data_col_number, data.shape[1]))

        # based on different data type to convert different logic.
        if not self.feature_list:
            # if we don't have any category features
            return data

        if isinstance(data, pd.DataFrame):
            converted_data = self.estimator.transform(data.iloc[:, self.feature_list].values).toarray()
            # HERE I have convert the columns features to index, so here should just remove these columns by name
            other_data = data.drop(df.columns[self.feature_list], axis=1)
            if self.keep_origin_feature:
                return np.concatenate([data.values, converted_data], axis=1)
            else:
                return np.concatenate([other_data.values, converted_data], axis=1)
        else:
            converted_data = self.estimator.transform(data[:, self.feature_list]).toarray()
            if self.keep_origin_feature:
                return np.concatenate([data, converted_data], axis=1)
            else:
                other_index = list(set(range(data.shape[1])) - set(self.feature_list))
                if not other_index:
                    # in case there isn't any other columns
                    return converted_data
                else:
                    return np.concatenate([data[:, other_index], converted_data], axis=1)

    @staticmethod
    def _get_category_features_indexes(x):
        """
        To get whole categorical features index.
        :param x:
        :return:
        """
        data_type = isinstance(x, pd.DataFrame)

        def _is_categorical_type(series):
            # after convert dataframe into array, if there is anything string, then others will be string too.
            # so we couldn't fit with array data type that has string type....
            return is_string_dtype(series) or is_categorical_dtype(series) or \
                   is_object_dtype(series) or is_bool_dtype(series) or is_integer_dtype(series)

        # with dataframe then return feature name list
        cate_feature_list = []
        if data_type:
            for feature_name in x.columns:
                if _is_categorical_type(x[feature_name]):
                    cate_feature_list.append(feature_name)
        else:
            # in case with just one columns with 1D
            if len(x.shape) == 1:
                if _is_categorical_type(x):
                    cate_feature_list.append(0)
            else:
                # loop for each column
                for i in range(x.shape[1]):
                    if _is_categorical_type(x[:, i]):
                        cate_feature_list.append(i)

        return cate_feature_list


if __name__ == '__main__':
    import numpy as np
    import random

    df = pd.DataFrame()

    df['x'] = np.random.uniform(0, 1, 100)
    df['y'] = [np.random.randint(4) for _ in range(len(df))]
    df['label'] = [random.choice(['a', 'b']) for _ in range(len(df))]

    onehot = OnehotEncoding(keep_origin_feature=False)

    onehot.fit(df)
    print(onehot.transform(df).shape)
