
from auto_ml.automl import ClassificationAutoML, FileLoad

# file_name = 'train.csv'
# file_path = r"C:\Users\guangqiiang.lu\Documents\lugq\code_for_future\auto_ml_pro\auto_ml\test"
# file_load = FileLoad(file_name, file_path=file_path, label_name='Survived')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x, y = load_iris(return_X_y=True)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)

auto_cl = ClassificationAutoML()
auto_cl.fit(x=xtrain, y=ytrain)

score = auto_cl.score(x=xtest, y=ytest)
pred = auto_cl.predict(x=xtest)
prob = auto_cl.predict_proba(x=xtest)

print("Get model score: ", score)
print("Get model prediction:", pred[:10])
print("Get model probability:", prob[:10])
