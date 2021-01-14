# -*- coding:utf-8 -*-
"""
This is used to make test data for whole project

@author: Guangqiang.lu
"""
import os
import pandas as pd
from auto_ml.utils.paths import get_root_path
from auto_ml.utils.backend_obj import Backend


backend = Backend()
root_path = get_root_path()
cur_path = os.path.join(root_path, 'test')


def get_training_data(return_df=False):
    df = pd.read_csv(os.path.join(cur_path, "train.csv"))

    if return_df:
        return df

    x = df.drop(['Survived'], axis=1).values
    y = df['Survived'].values
    x = x.copy(order='C')

    return x, y


def save_processing_data(data, dataset_name='process_tmp'):
    backend.save_dataset(data, dataset_name, model_file_path=False)
    print('Dataset has been saved into tmp folder.')


def load_processing_data(dataset_name='process_tmp'):
    """Load the processed data from disk"""
    dataset = backend.load_dataset(dataset_name)

    return dataset