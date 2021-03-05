
from auto_ml.automl import ClassificationAutoML, FileLoad

file_name = 'train.csv'
file_path = r"C:\Users\guangqiiang.lu\Documents\lugq\code_for_future\auto_ml_pro\auto_ml\test"
file_load = FileLoad(file_name, file_path=file_path, label_name='Survived')

auto_cl = ClassificationAutoML()
auto_cl.fit(file_load, val_split=0.2)

