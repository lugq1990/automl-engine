# -*- coding:utf-8 -*-
"""
First step should cover missing field logic.

@author: Guangqiang.lu
"""
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import KNNImputer, SimpleImputer
from auto_ml.backend_sklearn.preprocessing.processing_base import Process


class Impution(Process):
    def __init__(self, use_al_to_im=False):
        super(Impution, self).__init__()
        self.use_al_to_im = use_al_to_im

    def fit(self, data):
        """
        Impution logic happens here.
        I have to add a logic to process with different type of columns,
        like for numeric data could use distance based,
        for categorical, we could use most frequent items.
        #TODO have to check each column type and do separate processing then combine.
        :param data: contain missing field data
        :param use_al_to_im: Whether or not to use algorithm to impute data
        :return: fitted estimator
        """
        # Let's first try to split data into 2 parts with numeric and category
        numeric_index, category_index = self.get_col_data_type(data)
        # we have to save it for later transformation use
        self.numeric_index = numeric_index
        self.category_index = category_index

        # get different data
        numeric_data = data[:, numeric_index]
        category_data = data[:, category_index]

        try:
            if self.use_al_to_im:
                # for KNN, we could only use numerical data
                if category_index:
                    raise ValueError("For algorithm logic shouldn't contain category data")

                self.num_estimator = KNNImputer(n_neighbors=5)
            else:
                # Here just use univariate processing logic for missing values
                if numeric_index:
                    self.num_estimator = SimpleImputer()
                if category_index:
                    self.cate_estimator = SimpleImputer(missing_values='nan', strategy='most_frequent')

            # we have to check different data type to process.
            if numeric_index:
                self.num_estimator.fit(numeric_data)

            if category_index:
                self.cate_estimator.fit(category_data.astype(np.object))

        except Exception as e:
            raise Exception("When try to impute data with Imputer with error: {}".format(e))

    def transform(self, data):
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

    def fit_transform(self, data):
        """
        This have to overwrite parent func, as we need different processing logic
        :param data:
        :return:
        """
        # first we have to do fit logic, just like sklearn done.
        self.fit(data)

        return self.transform(data)

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
            while True:
                for j in range(len(sample)):
                    item = sample[j]
                    if _is_nan(item):
                        # if face nan value, go to next
                        continue
                    else:
                        # try to convert this data
                        try:
                            if isinstance(int(item), int):
                                numeric_col_index.append(i)
                        except:
                            cate_col_index.append(i)
                        break
                break

        return numeric_col_index, cate_col_index


if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)

    sample_data = np.array([["1", 2], ["3", 6], ["4", 8], [np.nan, 3], ["4", np.nan]])
    data_new = np.array([['good', 1], [np.nan, 2]])
    i = Impution()
    print(i.get_col_data_type(y))
    # n, c = i.get_col_data_type(x)
    # print(x[:, n])
    # print(i.fit_transform(data_new))

    i.fit(data_new)
    print(i.transform(data_new))
    print(i.fit_transform(sample_data))
