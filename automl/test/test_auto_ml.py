from  ClassificationAutoML, FileLoad
from sklearn.datasets import load_iris
import os
import numpy as np
import pandas as pd


auto_cl = ClassificationAutoML()

x, y = load_iris(return_X_y=True)

auto_cl.fit(x, y, val_split=.2)

# File load
data = np.concatenate([x, y[:, np.newaxis]], axis=1)
df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'label'])
df.to_csv("train.csv", index=False)

# If label in file also named with `label`, no need with set `label_name`
file_load = FileLoad('train.csv')

auto_cl.fit(file_load=file_load)