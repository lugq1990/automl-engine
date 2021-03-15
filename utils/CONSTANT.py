# -*- coding:utf-8 -*-
"""
This is whole constant value that we would use in the
whole project, as like what kind of problem that we
support, current we just support classification, regression.

@author: Guangqiang.lu
"""
BINARY_CLASSIFICATION = 0
MULTI_CLASS_CLASSIFICATION = 1
MULTI_LABEL_CLASSIFICATION = 2
REGRESSION = 3
CLUSTERING = 4
RECOMMENDATION = 5

# Here should confirm that there are 4 type tasks are supported
CLASSIFICTION_TASK = [BINARY_CLASSIFICATION, MULTI_CLASS_CLASSIFICATION, MULTI_LABEL_CLASSIFICATION]
REGRESSION_TASK = [REGRESSION]
CLUSTERING_TASK = [CLUSTERING]
RECOMMENDATION_TASK = [RECOMMENDATION]

WHOLE_TASK = CLASSIFICTION_TASK + REGRESSION_TASK + CLUSTERING_TASK + RECOMMENDATION_TASK

# Here make a dictionary to for recognize which task it is
TASK_TO_STRING = {BINARY_CLASSIFICATION: "binary",
                  MULTI_CLASS_CLASSIFICATION: "multiclass",
                  MULTI_LABEL_CLASSIFICATION: "multilabel",
                  REGRESSION: "regression",
                  CLUSTERING: "clustering",
                  RECOMMENDATION: "recommendation"}

# Here is to make string to task names
STRING_TO_TASK = {v: k for k, v in TASK_TO_STRING.items()}

# Here I just add a tmp folder path and model save path
import tempfile
import os

PROJECT_TMP_PATH = tempfile.mkdtemp()
# :TODO: in real prod, change this.
# PROJECT_TMP_PATH = "C:/Users/guangqiiang.lu/Documents/lugq/code_for_future/auto_ml_pro/auto_ml/tmp_folder"
TMP_FOLDER = os.path.join(PROJECT_TMP_PATH, "tmp")
OUTPUT_FOLDER = os.path.join(PROJECT_TMP_PATH, "models")

# Add validation split threshold
VALIDATION_THRESHOLD = 10000