"""Search a best neural network model with `kerastuner` based on data we have.

"""
import os
import traceback
# This won't work as check in source code that uses `print` to get info...

# import warnings
# warnings.simplefilter("ignore")
# # This only workable in linux
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# This is to filter the warning log
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, Conv1D, Activation
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch


from auto_ml.utils.paths import load_yaml_file
from auto_ml.utils.CONSTANT import OUTPUT_FOLDER
from auto_ml.utils.logger import create_logger


logger = create_logger(__file__)


hyper_yml_file_name = 'search_hypers.yml'
# Load hyperparameters
hyper_yml = load_yaml_file(hyper_yml_file_name)


# To build the search model based on the hyper yaml file
class SearchModel(HyperModel):
    def __init__(self, n_classes, algorithm_name='DNN', type_of_problem='classification', use_dropout=True):
        self.n_classes = n_classes
        self.algorithm_name = algorithm_name
        self.type_of_problem = type_of_problem
        self.use_dropout = use_dropout
        self.n_layers = hyper_yml[algorithm_name]['n_layers']
        self.n_units = hyper_yml[algorithm_name]['n_units']
        self.optimizers = hyper_yml['optimizers']
        self.hp = None
    
    def build(self, hp):
        raise NotImplementedError

    def _compile_model(self, model, optimizer_name, 
            loss_name='sparse_categorical_crossentropy', 
            metrics='accuracy', 
            type_of_problem='classification'):

        learning_optimizer = self._build_search_choice('learning_rate', self.optimizers[optimizer_name])
        
        if type_of_problem == 'regression':
            loss_name = 'mse'
            metrics = None

        # default to make with `Adam`
        optimizer = keras.optimizers.Adam(learning_optimizer)
        if optimizer_name == 'SGD':
            optimizer = keras.optimizers.SGD(learning_optimizer)
        
        compile_dict = {'optimizer': optimizer, "loss":loss_name, 'metrics': [metrics]}

        model.compile(**compile_dict)

        return model

    def _build_search_range(self, param_range, name=None):
        return self.hp.Int(name, param_range['min_value'], param_range['max_value'], param_range['step'])

    def _build_search_choice(self, name, param_range):
        return self.hp.Choice(name, param_range[name])



class DNNSearch(SearchModel):
    def __init__(self, n_classes, type_of_problem='classification', use_dropout=False):
        super().__init__(n_classes, algorithm_name='DNN', type_of_problem=type_of_problem, use_dropout=use_dropout)
        self.name = "DNN"

    def build(self, hp):
        self.hp = hp

        model = Sequential()

        for i in range(self._build_search_range(self.n_layers, 'n_layers')):
            model.add(Dense(units=self._build_search_range(self.n_units, 'n_units_' + str(i)), activation='relu'))
            if self.use_dropout:
                model.add(Dropout(0.5))
        
        model.add(Dense(self.n_classes))

        model = self._compile_model(model, 'Adam', type_of_problem=self.type_of_problem)
        return model


def save_keras_model(model, model_name, model_path=None, model_name_suffix='.h5'):
    if not model_name.endswith(model_name_suffix):
        model_name += model_name_suffix

    if not model_path:
        # let's try to save the model into default folder.
        model_path = OUTPUT_FOLDER

    try:
        model.save(os.path.join(model_path, model_name))

    except Exception as e:
        raise IOError("When try to save model: {} get error: {}".format(model_name, e))


class EvaluateNeuralModel:
    def __init__(self, model_list, x, y, algorithm_name='DNN'):
        self.model_list= model_list if isinstance(model_list, list) else [model_list]
        self.x = x
        self.y = y
        self.algorithm_name = algorithm_name
    
    def evaluate_models(self):
        self._check_models()

        score_list = []
        for model in self.model_list:
            score = model.evaluate(self.x, self.y)[1]
            # score will be with 6 digits
            score = round(score, 6)

            score_list.append(score)
        
        return score_list
    
    def save_models(self, model_path=None, model_name_suffix='.h5'):
        score_list = self.evaluate_models()

        if len(self.model_list) != len(score_list):
            raise ValueError("When to save model into disk, got score list is not equal to model list. \
                Model list number: {}, score list number:{}".format(len(self.model_list), len(score_list)))

        if not model_path:
            model_path = OUTPUT_FOLDER

        for model, model_score in zip(self.model_list, score_list):
            model_name = self.algorithm_name + '-' + str(model_score) + model_name_suffix
            try:
                model.save(os.path.join(model_path, model_name))
            except Exception as e:
                traceback.print_exc()
                raise IOError("When try to save model: {} into disk with error:{}".format(model, e))

    def _check_models(self):
        if not self.model_list:
            raise ValueError("Please provide at least one model! \
                Current model_list is {}".format(len(self.model_list)))



if __name__ == '__main__':
    model = DNNSearch(3)
    tuner = RandomSearch(model, objective='val_accuracy', 
        max_trials=10, executions_per_trial=3, 
        directory='./auto_ml/tmp_folder/tmp',  project_name='test')

    # tuner = NeuralModelRandomSearch(model, objective='val_accuracy', 
    #     max_trials=10, executions_per_trial=3, 
    #     directory='./auto_ml/tmp_folder/tmp',  project_name='test')

    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)

    print("Start to search")
    tuner.search(x, y, epochs=10, validation_split=.2)
    # tuner.save_best_models(x, y)

    print("Search step finished.")
    best_models = tuner.get_best_models(3)

    # print("Best model score:", [model.evaluate(x, y)[1] for model in best_models])
    # for model in best_models:
    #     model_name = "DNN" + str(model.evaluate(x, y)[1])
    #     model.save(os.path.join(OUTPUT_FOLDER, model_name) + '.h5')

    evaluate_model = EvaluateNeuralModel(best_models, x, y)
    print(evaluate_model.evaluate_models())
    evaluate_model.save_models()

