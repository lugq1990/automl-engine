# auto_ml

How to create a machine learning and deep learning models with just a few lines of code by just provide data, then framework
will get best trained models based on the data we have? We don't need to care about `feature engineering`, `model selection`, 
`model evaluation` and `model sink`, this is **automl** comes in.

This repository is based on **scikit-learn** and **TensorFlow** to create both machine learning models and nueral network models with few lines of code by just providing a training file, if there is a test file will be nicer to evaluate trained model without any bias, but if with just one file will also be fine.

Key features highlights:
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

# Get prediction based on best trained models
test_file_name = 'test.csv'
file_load_test = FileLoad(test_file_name, file_path=file_path, label_name='Survived')

pred = auto_cl.predict(file_load_test)
```

Then we could get whole trained models' evaluation score for each trained model score, we could get best trained model based on validation score if we would love to use trained model for production, one important thing is that these models are stored in local server, we could use them any time with RESTFul API calls.
![Evalution result](https://github.com/lugq1990/auto_ml/blob/master/test/diff_model_score.png)

If we want to use GCP cloud storage as a data source for train and test data, what needed is just get the service account file with proper authority, last is just provide with parameter: `service_account_name` and file local path: `service_account_file_path` to `FileLoad` object, then training will start automatically.

```python
service_account_name = "service_account.json"
service_account_file_path = r"C:\auto_ml\test"

file_load = FileLoad(file_name, file_path, label_name='Survived', 
    service_account_file_name=service_account_name, service_account_file_path=service_account_file_path)
```

If we have data in memory, we could also use memory objects to train, test and predict with `auto_ml` object, just like `scikit-learn`.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x, y = load_iris(return_X_y=True)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)

auto_cl = ClassificationAutoML()
auto_cl.fit(x=xtrain, y=ytrain)

score = auto_cl.score(x=xtest, y=ytest)
pred = auto_cl.predict(x=xtest)
prob = auto_cl.predict_proba(x=xtest)
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

Also supported with `Ensemble` logic to combine different models to build more powerful model by adding model diversity:
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


