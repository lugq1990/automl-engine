# -*- coding:utf-8 -*-
"""
This is backend object that we could use for whole file
transfer or file processing, we could just use this
class to do like model saving, model loading etc.

Here also use `Singleton` method to create the object.

author: Guangqiang.lu
"""
import pickle
import shutil
from numpy.lib.arraysetops import isin
import pandas as pd
import traceback

try:
    is_tf_support = True
    from tensorflow.keras.models import load_model as keras_load_model
except ImportError:
    is_tf_support = False

from auto_ml.utils.logger import create_logger
from auto_ml.utils.CONSTANT import *
from auto_ml.utils.paths import load_yaml_file


logger = create_logger(__file__)

hyper_yml_file_name = 'search_hypers.yml'
# Load hyperparametersload_model
hyper_yml = load_yaml_file(hyper_yml_file_name)


class Backend(object):
    """
    This the the common module that could be used for whole project,
    currently supported with save models files, load model from disk,
    should be added more here.
    """
    # This is used for Singleton.
    instance = None

    def __init__(self,
                 tmp_folder_name=None,
                 output_folder=None,
                 delete_tmp_folder_after_terminate=True,
                 delete_output_folder_after_terminate=True):
        # This is to ensure we could use same folder for whole project.
        # but one more thing is if we face any problems, then we will lose whole
        self.tmp_folder_name = TMP_FOLDER if tmp_folder_name is None else tmp_folder_name
        self.output_folder = OUTPUT_FOLDER if output_folder is None else output_folder

        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = delete_output_folder_after_terminate


        # I think when we init instance,then we should create the tmp folder first
        self._create_tmp_and_output_folder()

    def _create_tmp_and_output_folder(self):
        def __create_folder(folder_name):
            if not os.path.exists(folder_name):
                logger.info("Temp folder:{} don't exist, now try to create it.".format(folder_name))
                try:
                    os.makedirs(folder_name, exist_ok=True)
                except Exception as e:
                    logger.error("When create tmp and model folder with error: %s"
                                % traceback.format_exc())
                    raise IOError("When create tmp and model folder with error: %s" % e)
            else:
                logger.info("Folder: {} already exists.".format(folder_name))

        folder_list = [self.tmp_folder_name, self.output_folder]

        for folder in folder_list:
            __create_folder(folder)

    def delete_tmp_and_output_folder(self, delete_models=False):
        """As maybe we don't want to delete our trained model"""
        try:
            if self.delete_tmp_folder_after_terminate:
                shutil.rmtree(self.tmp_folder_name)

            if self.delete_output_folder_after_terminate and delete_models:
                shutil.rmtree(self.output_folder)
        except Exception as e:
            logger.exception("When try to remove temperate folder with error: %s" % e)

    def save_model(self, model, identifier):
        try:
            if not identifier.endswith(".pkl"):
                identifier += '.pkl'

            with open(os.path.join(self.output_folder, identifier), 'wb') as f:
                pickle.dump(model, f)

                logger.info("Model: {} has been saved into disk!".format(identifier))

        except Exception as e:
            logger.error("When try to save model: %s, face problem: %s" % (identifier, e))
            raise IOError("When try to save model: %s, face problem: %s" % (identifier, e))

    def load_model(self, identifier):
        """
        Get model instance object with just model name.
         Noted: `identifier` should contain with `extension`!

        :param identifier:
        :return:
        """
        # Here has to be changed for different platform
        default_neural_network_algorithms = hyper_yml['DefaultAlgorithms']
        identi_name = identifier.split('-')[0]
        if identi_name in default_neural_network_algorithms:
            # Keras model.
            return self.load_keras_model(identifier)

        # For sklearn models by using `pickle`
        try:
            model_file_path = os.path.join(self.output_folder, identifier)
            self._file_exists(model_file_path)

            with open(os.path.join(self.output_folder, identifier), 'rb') as f:
                model = pickle.load(f)

            return model
        except Exception as e:
            logger.error("When load %s model with error: %s." %
                         (identifier, traceback.format_exc()))
            raise IOError("When to load %s model, the file:%s not exist!" % (identifier, e))

    def load_keras_model(self, identifier):
        if not identifier.endswith('h5'):
            raise ValueError("To load keras model, model extension must end with `h5`")
        
        try:
            model_path = os.path.join(self.output_folder, identifier)
            
            self._file_exists(model_path)
            if not is_tf_support:
                raise RuntimeError("When try to use `TensorFlow` backend to load model, couldn't load module: `tensorflow`, please check!")

            model = keras_load_model(model_path)

            return model
        except IOError as e:
            logger.error("When to load keras model, get error: {}".format(e))
            raise IOError("When to load keras model, get error: {}".format(e))

    def _file_exists(self, file_path):
        """Check file exist or not based on path.

        Args:
            file_path (Str): File path to check.

        Raises:
            IOError: [description]
        """
        if not os.path.exists(file_path):
                raise IOError("File: {} not exist!".format(file_path))

    def load_models_by_identi_combined_with_model_name(self, identifiers):
        """
        As I don't want to change current logic with `load_model` to just
        return model instance, sometimes we also need the model name, so
        that we could also return (model_name, model_instance)
        :param identifiers:
        :return:
        """
        model_list = []

        if isinstance(identifiers, str):
            identifiers = [identifiers]

        for identi in identifiers:
            model_list.append((identi, self.load_model(identi)))

        return model_list

    def list_models(self, extension=['pkl', 'h5'], except_model_list=["processing_pipeline"]):
        """
        this is to list whole saved model in local disk,
        model name should be end with `pkl`, should be extended
        to other framework with other extension.
        :return: a list of model list
        """
        models_list = []

        # change this with keras models.
        if extension and isinstance(extension, list):
            for ex in extension:
                models_list.extend([x for x in os.listdir(self.output_folder) if x.endswith(ex)])
        # models_list = [x for x in os.listdir(self.output_folder) if x.endswith(extension)]
        
        if not models_list:
            logger.warning("There isn't any trained models in folder: {}".format(self.output_folder))
        

        # Here add logic to ensure we just add algorithm model list
        if except_model_list:
            models_list = [x for x in models_list if x.split('.')[0] not in except_model_list]

        return models_list

    def list_models_with_identifiers(self, identifiers, extension='pkl'):
        """
        this is to list whole models with identifiers satisfication.
        :param identifiers: a list of idenfifiers
        :param extension: what data type to include.
        :return: a model list with satisfied
        """
        models_list = [x for x in os.listdir(self.output_folder)
                       if x.endswith(extension) and x.split('.')[0] in identifiers]

        return models_list

    def load_models(self):
        """
        this is to load whole models already trained in disk.
        This should be algorithm models.
        :return: a list of trained model object.
        """
        model_obj_list = []
        for identifier in self.list_models():
            model_obj_list.append(self.load_model(identifier))
        return model_obj_list

    def load_models_combined_with_model_name(self):
        """
        To load models combined with `model name`, so later could use this
        models based on model score.
        :return: a list of models: [('LR-0.98.pkl', LogisticRegression-0.982323.pkl)]
        """
        model_with_name_list = []

        for identifier in self.list_models():
            model_with_name_list.extend(
                self.load_models_by_identi_combined_with_model_name(identifier))

        return model_with_name_list

    def load_models_with_identifiers(self, identifiers):
        """
        this is to load models with identifiers
        :param identifiers: a list of identifiers
        :return: a list of trained model object
        """
        model_obj_list = []
        for identifier in identifiers:
            model_obj_list.append(self.load_model(identifier))
        return model_obj_list

    def save_dataset(self, dataset, dataset_name, model_file_path=True):
        """
        As most of us have the structed data, so that I would love to
        store the data into disk with `csv` file, then we could use
        `pandas` to load it from the disk and convert it directly into
        DataFrame.
        :param dataset: dataset object to be saved
        :param dataset_name: what name for it, so that we could re-load it
        :param file_path: Where to save the data.
        :return:
        """
        if model_file_path:
            file_path = self.output_folder
        else:
            file_path = self.tmp_folder_name

        try:
            dataset_path = os.path.join(file_path, dataset_name + '.csv')
            if not isinstance(dataset, pd.DataFrame):
                dataset = pd.DataFrame(dataset, columns=[str(x) for x in range(1, dataset.shape[1] + 1)])

            dataset.to_csv(dataset_path, index=False, sep='|')
            logger.info("Dataset: {} has been saved into: {}".format(dataset_name, dataset_path))
        except IOError as e:
            raise IOError("When try to save dataset: {} get error: {}".format(dataset_name, e))

    def load_dataset(self, dataset_name, file_path=None, sep="|"):
        """
        To load csv data again from disk.
        :param dataset_name:
        :param file_path:
        :return:
        """
        if file_path is None:
            file_path = self.tmp_folder_name

        # we should ensure the dataset is endswith 'csv'
        if not dataset_name.endswith(".csv"):
            dataset_name += '.csv'

        if dataset_name not in os.listdir(file_path):
            raise IOError("When try to load dataset: {} from path:{}, "
                          "we couldn't find it!".format(dataset_name, file_path))

        try:
            dataset_path = os.path.join(file_path, dataset_name)
            dataset = pd.read_csv(dataset_path, sep=sep)

            return dataset
        except IOError as e:
            raise IOError("When try to load dataset: {} get error: {}".format(dataset_name, e))

    def clean_folder(self, folder_name=None):
        """
        To clean `folder_name` to ensure every time we could get a empty folder.
        :param folder_name: default is `models`
        :return:
        """
        if folder_name is None:
            # default is model path
            folder_name = self.output_folder
        
        if os.path.exists(folder_name):
            file_list = os.listdir(folder_name)
            if file_list:
                for file in file_list:
                    try:
                        os.remove(os.path.join(folder_name, file))
                    except IOError as e:
                        # in case if need with many trials couldn't delete these files, not good idea, so raise will be a solution
                        raise IOError("When try to remove file: {} in folder:{} get error: {}".format(file, folder_name, e))
        
    @classmethod
    def __new__(cls, *args, **kwargs):
        """
        `Singleton` is used to ensure there will be just one instance in
        whole running process.
        :param args:
        :param kwargs:
        :return:
        """
        if not cls.instance:
            # we should avoid to use __new__(cls, *args, **kwargs) as with error need to fix, just with __new__(cls) will be fine.
            # https://stackoverflow.com/questions/34777773/typeerror-object-takes-no-parameters-after-defining-new
            cls.instance = super(Backend, cls).__new__(cls)
            
        return cls.instance


if __name__ == "__main__":
    backend = Backend()
    print(backend.output_folder)
    print(backend.tmp_folder_name)

    # print(backend.load_models_combined_with_model_name())

    # # To test there is just one instance.
    # print(backend == Backend())
