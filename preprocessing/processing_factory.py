# -*- coding:utf-8 -*-
"""
This class is used as a `factory` to get instance that we need for the whole project using
`factory pattern`.

@author: Guangqiang.lu
"""
from auto_ml.preprocessing import \
    (onehotencoding,  standardization,  norlization,  minmax,  imputation,  feature_selection,  pca_reduction)


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
                processor_tuple.append((processor_name, imputation.Imputation()))
            elif processor_name == 'OnehotEncoding':
                processor_tuple.append((processor_name, onehotencoding.OnehotEncoding()))
            elif processor_name == 'Standard':
                processor_tuple.append((processor_name, standardization.Standard()))
            elif processor_name == 'Normalize':
                processor_tuple.append((processor_name, norlization.Normalize()))
            elif processor_name == 'MinMax':
                processor_tuple.append((processor_name, minmax.MinMax()))
            elif processor_name == 'FeatureSelect':
                processor_tuple.append((processor_name, feature_selection.FeatureSelect()))
            elif processor_name == 'PrincipalComponentAnalysis':
                processor_tuple.append((processor_name, pca_reduction.PrincipalComponentAnalysis()))

        return processor_tuple
