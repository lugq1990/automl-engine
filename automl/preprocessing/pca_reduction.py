# -*- coding:utf-8 -*-
"""
This is to use PCA to do feature reduction

@author: Guangqiang.lu
"""
import numpy as np
from sklearn.decomposition import PCA
from preprocessing.processing_base import Process


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


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x, y =load_iris(return_X_y=True)

    pca = PrincipalComponentAnalysis()
    pca.fit(x)

    print(pca.transform(x))
