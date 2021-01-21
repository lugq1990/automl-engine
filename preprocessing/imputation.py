# -*- coding:utf-8 -*-
"""
First step should cover missing field logic.
Here contain both algorithm based filling and frequency filling,
also if one columns contain too many missing data, then will
remove that column.

@author: Guangqiang.lu
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from auto_ml.preprocessing.processing_base import Process
from auto_ml.utils.data_rela import is_categorical_type


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

    # @staticmethod
    # def get_col_data_type(data):
    #     # with two different types of feature: `numeric` and `category`
    #     if len(data.shape) == 1:
    #         data = data.reshape(-1, 1)
    #     n_cols = data.shape[1]
    #     numeric_col_index = []
    #     cate_col_index = []
    #
    #     for i in range(data.shape[1]):
    #         if is_categorical_type(data[:, i]):
    #             cate_col_index.append(i)
    #         else:
    #             numeric_col_index.append(i)
    #
    #     return numeric_col_index, cate_col_index


if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)

    sample_data = np.array([["1", 2], ["3", 6], ["4", 8], [np.nan, 3], ["4", np.nan]])
    data_new = np.array([['good', 1], [np.nan, 2]])
    lable = [1, 2]
    i = Imputation()
    print(i.get_col_data_type(x))
    # n, c = i.get_col_data_type(x)
    # print(x[:, n])
    # print(i.fit_transform(data_new))

    i.fit(data_new, lable)
    print(i.transform(data_new))
    print(i.fit_transform(sample_data))
    print(i.name)

    print("*" * 30)
    from auto_ml.test.get_test_data import get_training_data, save_processing_data

    x, y = get_training_data()
    i.fit(x, y)
    x_new = i.transform(x)

    save_processing_data(x_new, 'imputation')
