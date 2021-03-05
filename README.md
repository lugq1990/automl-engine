# auto_ml

How to create a machine learning and deep learning models with just a few lines of code by just provide data, then framework
will get best trained models based on the data we have? We don't need to care about `feature engineering`, `model selection`, 
`model evaluation` and `model sink`, this is **automl** comes in.

This repository is based on **scikit-learn** and **TensorFlow** to create both machine learning models and nueral network models with few lines of code by just providing a training file, if there is a test file will be nicer to evaluate trained model without any bias, but if with just one file will also be fine.

Highlights:
 - `machine learning` and `neural network models` are supported.
 - `Automatically process` data with missing, unstable, categorical string combined data.
 - `Ensemble logic` to combine models to build more powerful models.
 - `Nueral network models search` with `kerastunner`.
 - `Cloud files` are supported like: `Cloud storage` for GCP or local files.
 - `Logging` different process information into one date file for future reference.

Training files are also supported stored in Cloud like: GCP's GCS with only a service account to interact with Cloud

Sample code to use `auto_ml` package.
```python

from auto_ml.automl import ClassificationAutoML, FileLoad

file_name = 'train.csv'
file_path = r"C:\auto_ml\test"  # Absolute path
file_load = FileLoad(file_name, file_path=file_path, label_name='Survived')

auto_cl = ClassificationAutoML()
auto_cl.fit(file_load, val_split=0.2)

```

Current supported algorithms:
 - Logistic Regression
 - Support vector machine
 - Gradient boosting tree
 - Random forest
 - Decision Tree
 - Adaboost Tree
 - K-neighbors
 - XGBoost
 - LightGBM
 - Deep nueral network
 - Convelutional nueral network
 - LSTM
 - etc.

Also supported with `ensemble` logic to combine different models to build more powerful model by adding model diversity:
 - Voting
 - Stacking

For raw data file, will try with some common pre-procesing steps to create dataset for algorithms, currently some pre-processing algorithms are supported:
 - Imputation with statistic analysis for continuous and categorical columns, also support with KNN imputaion for categorical columns.
 - Standarize with data standard data
 - Normalize 
 - OneHot Encoding for categorical columns
 - MinMax for continuous columns to avoid data volumn bias
 - PCA to demension reduction with threashold
 - Feature selection with variance or LinearRegression or ExtraTree


