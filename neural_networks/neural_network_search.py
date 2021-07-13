"""Search a best neural network model with `kerastuner` based on data we have.

"""
import os
import traceback
import time
import numpy as np
import string
import random
import shutil
import warnings
warnings.simplefilter("ignore")

from sklearn.model_selection import train_test_split

# This won't work as check `keras-tuner` source code that uses `print` to get info...
# # This only workable in linux
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# This is to filter tensorflow warning log
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, Conv1D, Activation
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch


from auto_ml.utils.paths import load_yaml_file
from auto_ml.utils.CONSTANT import OUTPUT_FOLDER, TMP_FOLDER
from auto_ml.utils.logger import create_logger
from auto_ml.utils.data_rela import get_num_classes_based_on_label, get_scorer_based_on_target, get_type_problem


logger = create_logger(__file__)


hyper_yml_file_name = 'search_hypers.yml'
# Load hyperparameters
hyper_yml = load_yaml_file(hyper_yml_file_name)
default_neural_algorithm_list = hyper_yml['DefaultAlgorithms']


# To build the search model based on the hyper yaml file
class SearchModel(HyperModel):
    def __init__(self, n_classes, algorithm_name='DNN', use_dropout=True):
        self.n_classes = n_classes
        self.algorithm_name = algorithm_name
        # self.type_of_problem = type_of_problem
        self.use_dropout = use_dropout
        self.n_layers = hyper_yml[algorithm_name]['n_layers']
        self.n_units = hyper_yml[algorithm_name]['n_units']
        self.optimizers = hyper_yml['optimizers']
        self.hp = None
    
    def build(self, hp):
        raise NotImplementedError

    def _compile_model(self, model, 
            optimizer_name, 
            loss_name='sparse_categorical_crossentropy', 
            metrics='accuracy', 
            type_of_problem='classification'):

        learning_optimizer = self._build_search_choice('learning_rate', self.optimizers[optimizer_name])
        
        if type_of_problem == 'regression':
            loss_name = 'mse'
            metrics = 'mse'

        # default to make with `Adam`
        optimizer = keras.optimizers.Adam(learning_optimizer)
        if optimizer_name == 'SGD':
            optimizer = keras.optimizers.SGD(learning_optimizer)
        
        compile_dict = {'optimizer': optimizer, "loss": loss_name, 'metrics': [metrics]}

        model.compile(**compile_dict)

        return model

    def _build_search_range(self, param_range, name=None):
        return self.hp.Int(name, param_range['min_value'], param_range['max_value'], param_range['step'])

    def _build_search_choice(self, name, param_range):
        return self.hp.Choice(name, param_range[name])



class DNNSearch(SearchModel):
    def __init__(self, n_classes, use_dropout=False):
        super().__init__(n_classes, algorithm_name='DNN', use_dropout=use_dropout)
        self.name = "DNN"

    def build(self, hp):
        self.hp = hp

        model = Sequential()

        for i in range(self._build_search_range(self.n_layers, 'n_layers')):
            model.add(Dense(units=self._build_search_range(self.n_units, 'n_units_' + str(i)), activation='relu'))
            if self.use_dropout:
                model.add(Dropout(0.5))
        
        # Based on different type, output activation should be different
        if self.n_classes >= 2:  
            activation = 'softmax'
        else:
            activation = None
        model.add(Dense(self.n_classes, activation=activation))

        # If `n_classes` is 1, then this is regression
        if self.n_classes == 1:
            type_of_problem = 'regression'
        else:
            type_of_problem = 'classification'

        model = self._compile_model(model, 'Adam', type_of_problem=type_of_problem)
        return model


class EvaluateNeuralModel:
    def __init__(self, model_list, x, y, algorithm_name='DNN', task_type='classification'):
        self.model_list = model_list if isinstance(model_list, list) else [model_list]
        self.x = x
        self.y = y
        self.algorithm_name = algorithm_name
        self.task_type = task_type
        self.score_list = []
    
    def evaluate_models(self):
        self._check_models()
        
        score_list = []
        for estimator in self.model_list:
            try:
                # score = estimator.evaluate(self.x, self.y)[1]
                scorer = get_scorer_based_on_target(self.task_type)
                task_type = get_type_problem(self.y)
                pred = estimator.predict(self.x)

                if task_type == 'classification':
                    # if this is classification problem, so change probability into prediction
                    pred = np.argmax(pred, axis=1)

                score = scorer(self.y, pred)

                # score will be with 6 digits
                mean_test_score = round(score, 6)
                
                estimator_train_name = estimator.name + "_" + str(score)

                self.score_list.append((estimator_train_name, estimator, mean_test_score))
                score_list.append(mean_test_score)
            except ValueError as e:
                raise ValueError("When try to evaluate model with self data get error: {}".format(e))
            
        return score_list
    
    def save_models(self, model_path, model_name_suffix='.h5'):
        score_list = self.evaluate_models()

        if len(self.model_list) != len(score_list):
            raise ValueError("When to save model into disk, got score list is not equal to model list. \
                Model list number: {}, score list number:{}".format(len(self.model_list), len(score_list)))

        if not model_path:
            model_path = OUTPUT_FOLDER

        for model, model_score in zip(self.model_list, score_list):
            model_name = self.algorithm_name + '_' + str(model_score) + model_name_suffix
            try:
                model.save(os.path.join(model_path, model_name))

                logger.info("Model: {} bas been save into folder: {}".format(model_name, model_path))
            except Exception as e:
                traceback.print_exc()
                raise IOError("When try to save model: {} into disk with error:{}".format(model, e))

    def _check_models(self):
        if not self.model_list:
            raise ValueError("Please provide at least one model! \
                Current model_list is {}".format(len(self.model_list)))


class NeuralNetworkFactory:
    def __init__(self):
        self.neural_networks_name_list = []
    
    @staticmethod
    def get_neural_model_instance(neural_networks_name_list, n_classes):
        """
        n_classes has to be provided as to init instance needs this.
        """
        if isinstance(neural_networks_name_list, str):
            neural_networks_name_list = [neural_networks_name_list]
        
        neural_networks_list = []
        for name in neural_networks_name_list:
            if name == 'DNN':
                neural_networks_list.append(DNNSearch(n_classes))
            elif name == 'CNN':
                pass

        return neural_networks_list


class NeuralModelSearch:
    """
    Main class for caller class to find best models.
    """
    def __init__(self, objective='val_accuracy', 
                    max_trials=1, 
                    executions_per_trial=1, 
                    directory=None, 
                    project_name=None, 
                    algorithm_list=None, 
                    tuning_algorithm='RandomSearch', 
                    num_best_models=5,
                    models_path=None, 
                    task_type='classification'):
        self.objective = objective
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.directory = directory if directory is not None else TMP_FOLDER
        # Here change the project name, as if we use same project, then won't re-fit just `reload`
        # make project name as random string.
        self.project_name = project_name if project_name is not None else self._generate_project_name() 
        
        # based on default hyper yaml file to define which mdoel to use.
        # default_keys = list(hyper_yml.keys())
        # default_keys.remove('optimizers')
        # Add parameter in yaml file will be better
        default_keys = default_neural_algorithm_list

        self.algorithm_list = default_keys if algorithm_list is None else algorithm_list
        self.tuning_algorithm = tuning_algorithm
        self.num_best_models = num_best_models
        
        # add with models_path for storing the models into path we want.
        self.models_path = models_path

        self.task_type = task_type

    def fit(self, x, y, epochs=10, val_x=None, val_y=None, validation_split=0.2, evaluate=True):
        """Search logic to find best model with support `classification` and `regression`

        Added with a evaluate score list like, so that we could make it with Grid search model: 
            estimator_train_name = estimator.name + "_" + str(mean_train_score)
            self.score_list.append((estimator_train_name, estimator, mean_test_score))
        Args:
            x ([type]): [description]
            y ([type]): [description]
            epochs ([type], optional): [description]. Defaults to 10.
            val_x ([type], optional): [description]. Defaults to None.
            val_y ([type], optional): [description]. Defaults to None.
            validation_split ([type], optional): [description]. Defaults to 0.2.
            evaluate ([type], optional): [description]. Defaults to True.
        """        
        tuner_list = self._get_search_tuner_list(x, y, task_type=self.task_type)

        # whether or not to evaluate should base on attribute
        if evaluate:
            if (not val_x and not val_y):
                train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=validation_split)
        
        # loop for tuner_list to try to search best model
        start_time = time.time()
        for i in range(len(tuner_list)):
            tuner = tuner_list[i]
            logger.info("Start to search neural network models.")
            if not evaluate:
                tuner.search(x, y, epochs=epochs, validation_split=validation_split)
            else:
                tuner.search(train_x, train_y, epochs=epochs, validation_data=(val_x, val_y))

            if evaluate:
                # get each search best models, evaluate and save it.
                best_models = tuner.get_best_models(self.num_best_models)
                
                if not best_models:
                    logger.warning("There is no best model found.")
                    return 
                
                # add attr to keep best trained models instances
                self.model_list = best_models
                self.evaluate_trained_models(best_models, val_x, val_y)

        # clean folder
        self._clean_search_space()

        logger.info("Whole fitting logic finished used {} seconds.".format(time.time() - start_time))

        return self

    def evaluate_trained_models(self, best_models, x, y):
        """After training, evaluate will get best trained model, get test score and save them into disk.

        Args:
            best_models ([type]): [description]
            x ([type]): [description]
            y ([type]): [description]
        """
        # get each search best models, evaluate and save it.
        evaluate_model = EvaluateNeuralModel(best_models, x, y, task_type=self.task_type)

        best_models_scores = evaluate_model.evaluate_models()
        self.score_list = evaluate_model.score_list

        logger.info("Get best scores are: [{}]".format('\t'.join([str(score) for score in best_models_scores])))

        logger.info("Start to save best trained nueral networks models into disk.")
        evaluate_model.save_models(self.models_path)
    
    def _get_search_tuner_list(self, x, y, task_type='classification'):
        """Based on different type of problem to construct many tunners.

        Args:
            x ([type]): [description]
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        if task_type == 'regression':
            num_classes = 1
            self.objective = 'val_loss'
        else:
            num_classes = get_num_classes_based_on_label(y)

        logger.info("Get {} type of problem with {} classes.".format(task_type, num_classes))

        logger.info("Start to get model instance for algorithms: [{}]".format('\t'.join(self.algorithm_list)))
        self.neural_networks_list = NeuralNetworkFactory.get_neural_model_instance(self.algorithm_list, num_classes)
        
        logger.info("Start to use search algorithm: {} to find models.".format(self.tuning_algorithm))
        
        tuner_list = []
        if self.tuning_algorithm == 'RandomSearch':
            for model in self.neural_networks_list:
                tuner_list.append(RandomSearch(model, 
                    objective=self.objective, 
                    max_trials=self.max_trials, 
                    executions_per_trial=self.executions_per_trial, 
                    directory=self.directory, 
                    project_name=self.project_name))
        else:
            pass
        
        return tuner_list

    @staticmethod
    def _generate_project_name(project_name_size=10, chars=string.ascii_uppercase + string.ascii_lowercase):
        """
        Random generate project name.
        """
        random_project_name = ''.join(random.choice(chars) for _ in range(project_name_size))
        
        return random_project_name

    def _clean_search_space(self):
        """Call only when `fit` logic finished."""
        try:
            logger.info("Try to clean serach model space folder: {}".format(self.project_name))

            search_project_path = os.path.join(self.directory, self.project_name)
            shutil.rmtree(search_project_path)
            logger.info("Folder: {} has been deleted!".format(self.project_name))

        except IOError as e:
            # if we couldn't delete this folder, then OK pass, but should log
            logger.error("To delete search project {} get error: {}".format(self.project_name, e))
            
        

if __name__ == '__main__':
    # model = DNNSearch(3)
    # tuner = RandomSearch(model, objective='val_accuracy', 
    #     max_trials=10, executions_per_trial=3, 
    #     directory='./auto_ml/tmp_folder/tmp',  project_name='test')

    # tuner = NeuralModelRandomSearch(model, objective='val_accuracy', 
    #     max_trials=10, executions_per_trial=3, 
    #     directory='./auto_ml/tmp_folder/tmp',  project_name='test')

    from sklearn.datasets import load_boston
    x, y = load_boston(return_X_y=True)

    # print("Start to search")
    # tuner.search(x, y, epochs=10, validation_split=.2)
    # # tuner.save_best_models(x, y)

    # print("Search step finished.")
    # best_models = tuner.get_best_models(3)

    # # print("Best model score:", [model.evaluate(x, y)[1] for model in best_models])
    # # for model in best_models:
    # #     model_name = "DNN" + str(model.evaluate(x, y)[1])
    # #     model.save(os.path.join(OUTPUT_FOLDER, model_name) + '.h5')

    # evaluate_model = EvaluateNeuralModel(best_models, x, y)
    # print(evaluate_model.evaluate_models())
    # evaluate_model.save_models()

    model_search = NeuralModelSearch()
    model_search.fit(x, y)

    # print("Get prediction: ", model_search.predict(x))
    # print("Get probability: ", model_search.predict_proba(x))
