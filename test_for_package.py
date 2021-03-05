
# ----- With file -----
from auto_ml.automl import ClassificationAutoML, FileLoad

file_name = 'train.csv'
file_path = r"C:\Users\guangqiiang.lu\Documents\lugq\code_for_future\auto_ml_pro\auto_ml\test"
file_load = FileLoad(file_name, file_path=file_path, label_name='Survived')

auto_cl = ClassificationAutoML()
auto_cl.fit(file_load, val_split=.2)

test_file_name = 'test.csv'
test_file_load = FileLoad(test_file_name, file_path=file_path, label_name='Survived')

# The reason to add this obj is for prediction, we don't need label in fact!
pred_file_load = FileLoad(test_file_name, file_path=file_path, label_name='Survived', use_for_pred=True)

pred = auto_cl.predict(pred_file_load)
print("Predict sample: ", pred[:5])
print('*'*30)
prob = auto_cl.predict_proba(pred_file_load)
print("Probability sample:", prob[:5])
print("*" * 30)
# score = auto_cl.score(test_file_load)

# ------ With in memory data ------

# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split

# x, y = load_digits(return_X_y=True)
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)

# models_path = r"C:\Users\guangqiiang.lu\Documents\lugq\code_for_future\auto_ml_pro\auto_ml\tmp_folder\tmp\models_folder_test"
# auto_cl = ClassificationAutoML(models_path=models_path)
# auto_cl.fit(x=xtrain, y=ytrain)

# score = auto_cl.score(x=xtest, y=ytest)
# pred = auto_cl.predict(x=xtest)
# prob = auto_cl.predict_proba(x=xtest)

# print("Get model score: ", score)
print("Get model prediction:", pred[:10])
print("Get model probability:", prob[:10])
