# -*- coding:utf-8 -*-
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Activation, LSTM, Bidirectional, GRU, RNN
from tensorflow.keras.models import Model

from .basic_net import BaseNet


class RnnNet(BaseNet):
    """This class implements recurrent neural networks including basic RNN, LSTM, GRU, Bidirectional LSTM.
        Arguments:
            n_classes(Integer): How many classes you want to classifier or use it for regression.
                Integer 2 is for binary problem, Integer >2 is for how many classes you have,
                Integert -1 is for regression problem.

            input_dim_1(Integer): First Dimension of your data sets. Noted, this is not batch-size dimension.

            input_dim_2(Integer): Second Dimension of your data sets.

            input_dim_3(Integer, optional): Third Dimension of your data sets. Noted, if data sets is just 2D,
                this parameter should not be given.

            n_layers(Integer, optional): How many recurrent layers wanted to be used.

            use_dropout(Boolean, optional): Whether or not to use Dropout for model training, avoiding model overfitting.

            drop_ratio(Double, optional): How many ratios to drop units in each layers.

            use_lstm(Boolean, optional): Whether or not to use LSTM units as basic recurrent.
                Default is True, use LSTM.

            use_bidirec(Boolean, optional): Whether or not to use bidirectional recurrent to train.
                Default is Flase, if set True, then if 'use_lstm' is also True, then use LSTM units as basic,
                if 'use_gru' is True, then use GRU units as basic, else just use RNN units as basic.

            use_gru(Boolean, optional): Whether or not to use GRU units as basic recurrent.

            rnn_units(Integer, optional): How many units to be used as recurrent units.

            use_dense(Boolean, optional): Whether or not to use Dense layers.
                Default is True

            dense_units(Integer, optional): How many units for building dense layers.

            use_batch(Boolean, optional): Whether or not to use BatchNormalization to normalize model weights.

            loss(String, optional): Which loss function to be used for model training.
                Default is None, use default loss function. For binary, use 'binary_crossentropy';
                For multi-class, use 'categorical_crossentropy'; For regression, use 'mse'.

            metrics(String, optional): Which metric to be used for model evaluation.

            optimizer(String, optional): Which optimizer to be used for model training optimizer.

            silence(Boolean, optional): Whether to print some useful information to console.
                Default is False, means print something to console.
    """

    def __init__(self, n_classes, input_dim_1, input_dim_2, input_dim_3=None, n_layers=3, use_dropout=True, drop_ratio=.5,
                 use_lstm=True, use_bidirec=False, use_gru=False, rnn_units=64, use_dense=True, dense_units=64, use_batch=True,
                 loss=None, metrics='accuracy', optimizer='rmsprop', silence=False):
        super().__init__()
        self.n_classes = n_classes
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.input_dim_3 = input_dim_3
        self.n_layers = n_layers
        self.use_dropout = use_dropout
        self.drop_ratio = drop_ratio
        self.use_lstm = use_lstm
        self.use_bidierc = use_bidirec
        self.use_gru = use_gru
        self.rnn_units = rnn_units
        self.use_dense = use_dense
        self.use_batch = use_batch
        self.dense_units = dense_units
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.silence = silence
        self.model = self._init_model()

    # this basic RNN block construction function.
    def _rnn_block(self, layers, name_index=None):
        if self.use_bidierc:
            if self.use_lstm:
                res = Bidirectional(LSTM(self.rnn_units, return_sequences=True, recurrent_dropout=self.drop_ratio))(layers)
            elif self.use_gru:
                res = Bidirectional(GRU(self.rnn_units, return_sequences=True, recurrent_dropout=self.drop_ratio))(layers)
            else:
                res = Bidirectional(RNN(self.rnn_units, return_sequences=True))(layers)

        elif self.use_gru:
            res = GRU(self.rnn_units, return_sequences=True,
                      recurrent_dropout=self.drop_ratio)(layers)

        elif self.use_lstm:
            res = LSTM(self.rnn_units, return_sequences=True,
                       recurrent_dropout=self.drop_ratio)(layers)
        else:
            res = RNN(self.rnn_units, return_sequences=True)(layers)

        if self.use_dropout:
            res = Dropout(self.drop_ratio)(res)

        return res

    def _init_model(self):
        inputs = Input(shape=(self.input_dim_1, self.input_dim_2))

        # No matter for LSTM, GRU, bidirection LSTM, final layer can not use 'return_sequences' output.
        for i in range(self.n_layers - 1):
            if i == 0:
                res = self._rnn_block(inputs, name_index=i)
            else:
                res = self._rnn_block(res, name_index=i)

        # final LSTM layer
        if self.use_bidierc:
            res = Bidirectional(LSTM(self.rnn_units))(res)
        elif self.use_gru:
            res = GRU(self.rnn_units)(res)
        elif self.use_lstm:
            res = LSTM(self.rnn_units)(res)
        else:
            res = RNN(self.rnn_units)(res)

        # whether or not to use Dense layer
        if self.use_dense:
            res = Dense(self.dense_units)(res)
            if self.use_batch:
                res = BatchNormalization()(res)
            res = Activation('relu')(res)
            if self.use_dropout:
                res = Dropout(self.drop_ratio)(res)

                # this is method private function to check whether or not loss is not given, then use default loss

        def _check_loss(model, loss, metrics, optimizer):
            if loss is not None:
                model.compile(loss=loss, metrics=[metrics], optimizer=optimizer)
            return model

        if self.n_classes == 2:  # this is binary class problem.
            out = Dense(self.n_classes, activation='sigmoid')(res)
            model = Model(inputs, out)
            if self.loss is None:
                model.compile(loss='binary_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
            else:
                _check_loss(model, self.loss, self.metrics, self.optimizer)
        elif self.n_classes >= 2:  # this is multiclass problem.
            out = Dense(self.n_classes, activation='softmax')(res)
            model = Model(inputs, out)
            if self.loss is None:
                model.compile(loss='categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
            else:
                _check_loss(model, self.loss, self.metrics, self.optimizer)
        elif self.n_classes == -1:  # this is regression problem
            out = Dense(1)(res)
            model = Model(inputs, out)
            if self.loss is None:
                model.compile(loss='mse', metrics=[self.metrics], optimizer=self.optimizer)
            else:
                _check_loss(model, self.loss, self.metrics, self.optimizer)
        else:
            raise AttributeError("Parameter 'n_classes' should be -1, 2 or up 2!")

        if not self.silence:
            print('Model structure summary:')
            model.summary()

        return model

