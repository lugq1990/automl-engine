# -*- coding:utf-8 -*-
"""
This is main class that is used for whole processing logic for sklearn.

@author: Guangqiang.lu
"""
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


from .utils.data_rela import check_label, is_categorical_type
from .utils.CONSTANT import *



class ProcessingFactory:
    @staticmethod
    def get_processor_list(processor_name_list):
        """
        Simple factory.
        
        As sklearn pipeline is a list of tuple(name,  transform, so here will suit for it.
        :param processor_name_list: which algorithms to include.
        :return:
        """
        processor_tuple = []

        if isinstance(processor_name_list,  str):
            processor_name_list = [processor_name_list]

        for processor_name in processor_name_list:
            if processor_name == 'Imputation':
                processor_tuple.append((processor_name, Imputation()))
            elif processor_name == 'OnehotEncoding':
                processor_tuple.append((processor_name, OnehotEncoding()))
            elif processor_name == 'Standard':
                processor_tuple.append((processor_name,Standard()))
            elif processor_name == 'Normalize':
                processor_tuple.append((processor_name, Normalize()))
            elif processor_name == 'MinMax':
                processor_tuple.append((processor_name, MinMax()))
            elif processor_name == 'FeatureSelect':
                processor_tuple.append((processor_name, FeatureSelect()))
            elif processor_name == 'PrincipalComponentAnalysis':
                processor_tuple.append((processor_name, PrincipalComponentAnalysis()))

        return processor_tuple


class Process(TransformerMixin):
    """
    This is whole class that is used for pre-processing logic, just give a direction.
    Here for init function that we want to get just class name for later process.
    """
    def __init__(self):
        self.name = self.__class__.__name__
        self.estimator = None

    def fit(self, x, y=None):
        """
        For whole processing logic should provide with label, so that even we don't use
        it, we could just follow sklearn logic.
        :param data:
        :param label:
        :return:
        """
        self.estimator.fit(x, y=y)

    def fit_transform(self, data, y=None):
        """
        Make parent logic just like sklearn.
        :param data:
        :return:
        """
        self.fit(data, y=y)
        return self.transform(data, y=y)

    def transform(self, data, y=None):
        return self.estimator.transform(data)




class Standard(Process):
    def __init__(self):
        super(Standard, self).__init__()
        self.estimator = StandardScaler()



cols_keep_ratio = .8


class PrincipalComponentAnalysis(Process):
    def __init__(self, n_components=None, selection_ratio=.90):
        """
        PCA for data decomposition with PCA
        :param n_components: how many new components to keep
        :param selection_ratio: how much information to keep to get fewer columns
        """
        super(PrincipalComponentAnalysis, self).__init__()
        self.estimator = PCA(n_components=n_components)
        self.selection_ratio = selection_ratio

    def transform(self, data, y=None):
        """
        Here I want to do feature decomposition based on pca score to reduce to less feature

        :param data:
        :return:
        """
        # first let me check the estimator is fitted
        if not hasattr(self.estimator, 'mean_'):
            raise Exception("PCA model hasn't been fitted")

        ratio_list = self.estimator.explained_variance_ratio_
        ratio_cum = np.cumsum(ratio_list)
        n_feature_selected = sum(ratio_cum < self.selection_ratio)
        if (n_feature_selected / data.shape[1]) < cols_keep_ratio:
            # we don't want to get so less features
            n_feature_selected = int(cols_keep_ratio * data.shape[1])

        return self.estimator.transform(data)[:, :n_feature_selected]




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




class Normalize(Process):
    def __init__(self):
        super(Normalize, self).__init__()
        self.estimator = Normalizer()





class MinMax(Process):
    def __init__(self):
        super(MinMax, self).__init__()
        self.estimator = MinMaxScaler()
        



class Imputation(Process):
    def __init__(self, use_al_to_im=False, threshold=.5):
        super(Imputation, self).__init__()
        self.use_al_to_im = use_al_to_im
        self.threshold = threshold

    def fit(self, data, y=None):
        """
        Imputation logic happens here.
        I have to add a logic to process with different type of columns,
        like for numeric data could use distance based,
        for categorical, we could use most frequent items.

        :param data: contain missing field data
        :param use_al_to_im: Whether or not to use algorithm to impute data
        :return: fitted estimator
        """
        # before we do any thing, should remove too many mssing cols
        self._remove_cols_up_to_threshold(data)

        # Let's first try to split data into 2 parts with numeric and category
        numeric_index, category_index = self.get_col_data_type(data)
        # we have to save it for later transformation use
        self.numeric_index = numeric_index
        self.category_index = category_index

        # get different data, there maybe without numeric or category index
        if self.numeric_index:
            numeric_data = data[:, self.numeric_index]
        if self.category_index:
            category_data = data[:, self.category_index]

        try:
            if self.use_al_to_im:
                # for KNN, we could only use numerical data
                if category_index:
                    raise ValueError("For algorithm logic shouldn't contain category data")

                self.num_estimator = KNNImputer(n_neighbors=5)
            else:
                # Here just use univariate processing logic for missing values
                if self.numeric_index:
                    self.num_estimator = SimpleImputer()
                if self.category_index:
                    self.cate_estimator = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

            # we have to check different data type to process.
            if self.numeric_index:
                self.num_estimator.fit(numeric_data)

            if self.category_index:
                self.cate_estimator.fit(category_data.astype(np.object))

        except Exception as e:
            raise Exception("When try to impute data with Imputer with error: {}".format(e))

    def transform(self, data, y=None):
        # # before we do anything, first should keep the data that we want based on training data
        # data = self._transform_data_without_col_over_threshold(data)

        # here just to according to different type data to transform data and combine them
        if self.numeric_index:
            numeric_data = self.num_estimator.transform(data[:, self.numeric_index])

        if self.category_index:
            # we have to add `.astype(np.object)`, as imputation only work with object type
            cate_data = self.cate_estimator.transform(data[:, self.category_index].astype(np.object))

        # we could just drop the order of the data, then just combine them here
        # could do better with origin order
        if self.numeric_index and self.category_index:
            return np.concatenate([numeric_data, cate_data], axis=1)
        elif self.numeric_index:
            return numeric_data
        else:
            return cate_data

    def fit_transform(self, data, y=None):
        """
        This have to overwrite parent func, as we need different processing logic
        :param data:
        :return:
        """
        # first we have to do fit logic, just like sklearn done.
        self.fit(data, y=None)

        return self.transform(data, y=None)

    def _transform_data_without_col_over_threshold(self, data):
        """
        To keep the cols that are kept in the training process.
        :param data:
        :return:
        """
        if isinstance(data, pd.DataFrame):
            data = data[self.keep_cols]
        else:
            data = data[:, self.keep_cols]

        return data

    def _remove_cols_up_to_threshold(self, data):
        """
        As if there are too many missing value, we don't need it, just remove it

        Have to store the column that we want to keep for `transform` use case to remove
        the column that is over `threshold` NAN values.
        :param threshold: default is 0.5
        :return:
        """
        if isinstance(data, pd.DataFrame):
            null_ratio_series = data.isnull().sum() / len(data)
            self.keep_cols = list(null_ratio_series[null_ratio_series < self.threshold].index)
            data = data[self.keep_cols]
        else:
            null_ratio_series = pd.isnull(data).sum(axis=0) / len(data)
            self.keep_cols = np.array(range(data.shape[1]))[null_ratio_series < self.threshold]
            data = data[:, self.keep_cols]

        return data

    @staticmethod
    def get_col_data_type(data):
        """
        To get each column data type with some missing values
        :param data:
        :return:
        """
        # with two different types of feature: `numeric` and `category`
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        n_cols = data.shape[1]
        numeric_col_index = []
        cate_col_index = []

        # just define a func to check value is nan or not
        def _is_nan(item):
            return item != item

        for i in range(n_cols):
            sample = data[:, i]
            # currently logic to try to get each value
            # todo: Must be changed, if there is column first as number, the other with string will get error.
            try:
                if all([isinstance(int(s), int) for s in sample if not _is_nan(s)]):
                    # if and only if whole value are number and the value is not NAN
                    numeric_col_index.append(i)
            except:
                cate_col_index.append(i)

            # while True:
            #     for j in range(len(sample)):
            #         item = sample[j]
            #         if _is_nan(item):
            #             # if face nan value, go to next
            #             continue
            #         else:
            #             # try to convert this data
            #             try:
            #                 if isinstance(int(item), int):
            #                     numeric_col_index.append(i)
            #             except:
            #                 cate_col_index.append(i)
            #             break
            #     break

        return numeric_col_index, cate_col_index





class FeatureSelect(Process):
    def __init__(self, simple_select=True, tree_select=False):
        super(FeatureSelect, self).__init__()
        self.simple_select = simple_select
        self.tree_select = tree_select

    def fit(self, data, y=None):
        """
        Also support with algorithm based feature selection
        :param data: data to process
        :param label: label data if need with algorithm trained based
        :return:
        """
        if self.simple_select:
            self.estimator = VarianceThreshold()
            self.estimator.fit(data)
        else:
            # if we want to use algorithms based feature extraction, currently will use LinearRegression
            if y is None:
                raise ValueError("When want to use Algorithm, label data should be provided!")

            # before next step, we should ensure task should be classification
            label_type = check_label(y)
            if label_type not in CLASSIFICTION_TASK:
                raise ValueError("When we want to use model selection logic, task type should just be classification.")

            if self.tree_select:
                model = ExtraTreesClassifier(n_estimators=50).fit(data, y)
            else:
                model = LinearRegression().fit(data, y)

            self.estimator = SelectFromModel(model, prefit=True)