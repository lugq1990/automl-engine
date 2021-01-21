# -*- coding:utf-8 -*-
"""
This is used to process with categorical data type to convert it into onehot type

@author: Guangqiang.lu
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from auto_ml.preprocessing.processing_base import Process
from auto_ml.utils.data_rela import is_categorical_type


class OnehotEncoding(Process):
    def __init__(self, keep_origin_feature=False,
                 except_feature_indexes=None,
                 except_feature_names_list=None,
                 drop_ratio=.2,
                 max_categoric_number=30):
        """
        In case there will be numpy array or pandas DataFrame data type,
        so that we could try to use both of them types.
        :param keep_origin_feature:
            when to transform, to keep original feature or not.
        :param except_feature_indexes:
            array column indexes
        :param except_feature_names_list:
            some features doesn't need to convert even they could.
        :param drop_ratio:
            if there are too many categorical features, so we should make a threshould
            that if there are categorical feature over `drop_ratio`, then we should just
            drop this feature.
        :param max_categoric_number:
            in case there are too many categorical values, even with the `drop_ratio`,
            we still get too many features, this is not we want.
        """
        super(OnehotEncoding, self).__init__()
        self.keep_origin_feature = keep_origin_feature
        self.except_feature_indexes = except_feature_indexes
        self.except_feature_names_list = except_feature_names_list
        self.drop_ratio = drop_ratio
        self.max_categoric_number = max_categoric_number

    def fit(self, x, y=None):
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
        # remove this, just to use the column's name
        # if self.data_type:
        #     feature_name_list = list(x.columns)
        #     self.feature_list = [feature_name_list.index(x) for x in self.feature_list]

        # Here I can't use pandas, as if we need to do same logic with test data, we need to store the model.
        # return pd.get_dummies(x, prefix=feature_list)

        self.estimator = OneHotEncoder(handle_unknown='ignore')
        x = self._get_feature_data_with_threshold(x)

        if not self.feature_list:
            # in case that we don't get any of category dataset,
            # then for the transform should be passed with original data
            return

        self.estimator.fit(x)
        return self

    def transform(self, x, y=None):
        """
        Should try to keep the data type with same type like array or pandas,
        so even if we fit with pandas, but we still could use array data type to do transform.
        Originally I want to keep the data type also into dataframe if provided dataframe,
        but that's useless, so there just to make it into a array.
        :param data: data could be DataFrame or array type could be fine.
        :return: array type
        """
        # at least we should ensure with same column numbers
        if x.shape[1] != self.data_col_number:
            raise ValueError("When to do transform logic, should provide same dimension, "
                             "fitted data is: {}, current is {}".format(self.data_col_number, x.shape[1]))

        # based on different data type to convert different logic.
        if not self.feature_list:
            # if we don't have any category features
            return x

        # We will combine `OneHot` result and `original numerical data` into a new dataset
        # based on `DataFrame` or `Array`
        if isinstance(x, pd.DataFrame):
            # we have store the feature name list with real name
            converted_data = self.estimator.transform(x[self.feature_list].values).toarray()

            other_data = x.drop(self.feature_list, axis=1)
            # the other data should also drop the feature dropped by OneHot logic
            if self.drop_feature_list:
                other_data.drop(self.drop_feature_list, axis=1, inplace=True)

            if self.keep_origin_feature:
                return np.concatenate([x.values, converted_data], axis=1)
            else:
                return np.concatenate([other_data.values, converted_data], axis=1)
        else:
            converted_data = self.estimator.transform(x[:, self.feature_list]).toarray()

            if self.keep_origin_feature:
                return np.concatenate([x, converted_data], axis=1)
            else:
                other_index = list(set(range(x.shape[1])) - set(self.feature_list) - set(self.drop_feature_list))
                if not other_index:
                    # in case there isn't any other columns
                    return converted_data
                else:
                    return np.concatenate([x[:, other_index], converted_data], axis=1)

    @staticmethod
    def _get_category_features_indexes(x):
        """
        To get whole categorical features index.
        :param x:
        :return:
        """
        data_type = isinstance(x, pd.DataFrame)

        # def _is_categorical_type(series):
        #     # after convert dataframe into array, if there is anything string, then others will be string too.
        #     # so we couldn't fit with array data type that has string type....
        #     return is_string_dtype(series) or is_categorical_dtype(series) or \
        #            is_object_dtype(series) or is_bool_dtype(series) or is_integer_dtype(series)

        # with dataframe then return feature name list
        cate_feature_list = []
        if data_type:
            for feature_name in x.columns:
                if is_categorical_type(x[feature_name]):
                    cate_feature_list.append(feature_name)
        else:
            # in case with just one columns with 1D
            if len(x.shape) == 1:
                if is_categorical_type(x):
                    cate_feature_list.append(0)
            else:
                # loop for each column
                for i in range(x.shape[1]):
                    if is_categorical_type(x[:, i]):
                        cate_feature_list.append(i)

        return cate_feature_list

    def _get_feature_data_with_threshold(self, x):
        if self.feature_list:
            # drop the feature number over some threshold or over some number
            self._process_feature_list_under_threshold(x)

            if self.data_type:
                # if this is a dataframe
                x = x[self.feature_list]
                # convert it into a array
                x = x.values
            else:
                x = x[:, self.feature_list]
                if len(self.feature_list) == 1:
                    x = x.reshape(-1, 1)
            return x

    def _process_feature_list_under_threshold(self, x):
        """
        To change the feature list that is under the threshold or satisfy requirement.
        :param x:
        :return:
        """
        if self.data_type:
            # to decide the feature list that we want
            # convert to type: `str` to avoid error: TypeError: '<' not supported between instances of 'str' and 'float'
            keep_feature_list = [self._keep_feature_under_threshold(x[fea].astype('str')) \
                                 for fea in self.feature_list]
        else:
            keep_feature_list = [self._keep_feature_under_threshold(x[:, fea]) \
                                 for fea in self.feature_list]

        # Don't change it right now, otherwise the later step will fail
        new_feature_list = [fea for fea, keep in zip(self.feature_list, keep_feature_list) if keep is True]
        # here should store not used feature, as for the `transform` action, if we drop these feature names
        # then if we need to get the rest of the data, we will combine the feature we don't want
        self.drop_feature_list = [fea for fea, keep in zip(self.feature_list, keep_feature_list) if keep is False]

        self.feature_list = new_feature_list

    def _keep_feature_under_threshold(self, data):
        """
        Based on each feature to decide to keep the feature or not.
        :param data:
        :return:
        """
        categorical_feature_num = len(np.unique(data))

        if categorical_feature_num < self.max_categoric_number:
            return True
        return False


if __name__ == '__main__':
    # import numpy as np
    # import random
    #
    # df = pd.DataFrame()
    #
    # df['x'] = np.random.uniform(0, 1, 100)
    # df['y'] = [np.random.randint(4) for _ in range(len(df))]
    # df['label'] = [random.choice(['a', 'b']) for _ in range(len(df))]
    from auto_ml.test.get_test_data import save_processing_data, load_processing_data

    df = load_processing_data('imputation.csv')

    onehot = OnehotEncoding(keep_origin_feature=False)

    onehot.fit(df)
    onehot_res = onehot.transform(df)

    save_processing_data(onehot_res, 'onehot')


