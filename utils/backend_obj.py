# -*- coding:utf-8 -*-
"""
This is backend object that we could use for whole file
transfer or file processing, we could just use this
class to do like model saving, model loading etc.

author: Guangqiang.lu
"""
import pickle
import tempfile
import os
import shutil
import traceback
from auto_ml.utils.logger import create_logger


logger = create_logger()


class Backend(object):
    """
    This the the common module that could be used for whole project,
    currently supported with save models files, load model from disk,
    should be added more here.
    """
    def __init__(self,
                 tmp_folder_name=None,
                 output_folder=None,
                 delete_tmp_folder_after_terminate=True,
                 delete_output_folder_after_terminate=True):
        self.tmp_folder_name = os.path.join(tempfile.gettempdir(), 'tmp') \
            if tmp_folder_name is None else tmp_folder_name

        self.output_folder = os.path.join(tempfile.gettempdir(), 'models') \
            if output_folder is None else output_folder

        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = delete_output_folder_after_terminate

        # I think when we init instance,then we should create the tmp folder first
        self._create_tmp_and_output_folder()

    def _create_tmp_and_output_folder(self):
        if not (os.path.exists(self.tmp_folder_name) or os.path.exists(self.output_folder)):
            try:
                os.makedirs(self.tmp_folder_name)
                os.makedirs(self.output_folder)
            except Exception as e:
                logger.error("When create tmp and model folder with error: %s"
                             % traceback.format_exc())
                raise IOError("When create tmp and model folder with error: %s" % e)

    def delete_tmp_and_output_folder(self):
        try:
            if self.delete_tmp_folder_after_terminate:
                shutil.rmtree(self.tmp_folder_name)

            if self.delete_output_folder_after_terminate:
                shutil.rmtree(self.output_folder)
        except Exception as e:
            logger.exception("When try to remove temperate folder with error: %s" % e)

    def save_model(self, model, identifier):
        try:
            with open(os.path.join(self.output_folder, identifier + '.pkl'), 'wb') as f:
                pickle.dump(model, f)

        except Exception as e:
           raise IOError("When try to save model: %s, face problem: %s" % (identifier, e))

    def load_model(self, identifier):
        try:
            with open(os.path.join(self.output_folder, identifier + '.pkl'), 'rb') as f:
                model = pickle.load(f)

            return model
        except Exception as e:
            logger.error("When load %s model with error: %s" %
                         (identifier, traceback.format_exc()))
            raise IOError("When to load %s model, face problem:%s" % (identifier, e))

    def list_models(self, extension='pkl'):
        """
        this is to list whole saved model in local disk,
        model name should be end with `pkl`, should be extended
        to other framework with other extension.
        :return: a list of model list
        """
        models_list = [x for x in os.listdir(self.output_folder) if x.endswith(extension)]
        return models_list

    def list_models_with_identifiers(self, identifiers, extension='pkl'):
        """
        this is to list whole models with identifiers satisfication.
        :param identifiers: a list of idenfifiers
        :return: a model list with satisfied
        """
        models_list = [x for x in os.listdir(self.output_folder)
                       if x.endswith(extension) and x.split('.')[0] in identifiers]

        return models_list

    def load_models(self):
        """
        this is to load whole models already trained in disk
        :return: a list of trained model object.
        """
        model_obj_list = []
        for identifier in self.list_models():
            model_obj_list.append(self.load_model(identifier))
        return model_obj_list

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


if __name__ == "__main__":
    backend = Backend()
    print(backend.output_folder)
