# -*- coding:utf-8 -*-
"""
Parent for whole `networks` for this project that we could use for sub-class

@author: Guangqiang.lu
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from utils.backend_obj import Backend
from utils.logger import create_logger

logger = create_logger(__file__)


class BaseNet(object):
    def __init__(self):
        self.backend = Backend()
        self.model = None

    def fit(self, data, label, epochs=100, batch_size=128, callback=None, patience=10):
        """
        Model fitting function, here this function will split data to be train and validation data sets.
        Model training will use train data, and will also evaluate after training model accuracy.
        :param model:
        :param data:
        :param label:
        :param n_classes:
        :param epochs:
        :param batch_size:
        :param callback:
        :param patience:
            how many epochs to wait for the model to train.
        :param silence:
        :return:
        """
        label = self._check_label(label)

        xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)

        if callback is None:
            callback = EarlyStopping(monitor='val_loss', patience=patience, min_delta=1e-4)

        logger.info("Start to `train` neural network model based on training data!")
        self.his = self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size,
                            validation_data=(xvalidate, yvalidate), callbacks=[callback], verbose=1)

        val_acc = self.model.evaluate(xvalidate, yvalidate, batch_size=1024)[1]
        logger.info('After training, model evaluate on validate data accuracy = {:.6f}%'.format(val_acc * 100))

        # As we need to save trained model into disk, we need the validation score combined with class name...so store
        # After the model is fitted, then let's plot the result, so we could get a better understanding.
        # self.plot_acc()

        return self.his

    def predict(self, data, batch_size=1024):
        """
        Get max index as prediction
        :param data:
        :param batch_size:
        :param step:
        :return:
        """
        prob = self.predict_proba(data, batch_size)

        if len(prob.shape) > 1 and len(prob) > 0:
            # just to ensure that we have get proper probability
            pred = np.argmax(prob, axis=1)
        else:
            raise ValueError("When to use neural network's `predict` func get error!")

        return pred

    def predict_proba(self, data, batch_size=1024):
        if not hasattr(self.model, 'predict'):
            # by default, keras will use `predict` to return the probability
            logger.error("Neural network's model hasn't been fitted, so please fit model first!")
            raise NotFittedError("Neural network's model hasn't been fitted, so please fit model first!")

        prob = self.model.predict(data, batch_size=batch_size)

        return prob

    def score(self, data, label, batch_size=1024):
        label = self._check_label(label)

        eval_acc = self.model.evaluate(data, label, batch_size=batch_size)[1]

        # To keep score with 6 digits to keep step with the whole framework.
        eval_acc = round(eval_acc, 6)
        logger.info('Model evaluate on `test data` accuracy = {:.6f}%'.format(eval_acc))

        return eval_acc

    def save(self, model_name, path=None):
        """
        Save model into disk.
        :param model_name:
        :param path:
        :return:
        """
        if path is None:
            # If path is not given, save model to default model path
            path = self.backend.output_folder

        if not model_name.endswith(".h5"):
            model_name += '.h5'

        self.model.save(os.path.join(path, model_name))
        logger.info('Model have been saved to disk: {}'.format(os.path.join(path, model_name)))

    def plot_acc(self, plot_acc=True, plot_loss=True, figsize=(8, 6)):
        """
        Currently is for `classification`.
        :param plot_acc:
        :param plot_loss:
        :param figsize:
        :return:
        """
        style.use('ggplot')

        if plot_acc:
            fig_1, ax_1 = plt.subplots(1, 1, figsize=figsize)
            ax_1.plot(self.his.history['accuracy'], label='Train Accuracy')
            ax_1.plot(self.his.history['accuracy'], label='Validation Accuracy')
            ax_1.set_title('Train & Validation Accuracy curve')
            ax_1.set_xlabel('Epochs')
            ax_1.set_ylabel('Accuracy score')
            plt.legend()

            plt.show()

        if plot_loss:
            fig_2, ax_2 = plt.subplots(1, 1, figsize=figsize)
            ax_2.plot(self.his.history['loss'], label='Train Loss')
            ax_2.plot(self.his.history['val_loss'], label='Validation Loss')
            ax_2.set_title('Train & Validation Loss curve')
            ax_2.set_xlabel('Epochs')
            ax_2.set_ylabel('Loss score')
            plt.legend()

            plt.show()

    # This is used for converting 1D label to nD one-hot label
    @staticmethod
    def _check_label(label):
        """
        Only use for `classification` problem.
        :param label:
        :return:
        """
        if len(label.shape) == 1:
            return tf.keras.utils.to_categorical(label, num_classes=len(np.unique(label)))

        return label
