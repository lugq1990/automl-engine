# -*- coding:utf-8 -*-
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Input, BatchNormalization,\
    Conv2D, MaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, Activation, Flatten
from tensorflow.keras.models import Model

from auto_ml.nueral_networks.basic_net import BaseNet


class BasicCnnNet(BaseNet):
    """This is a basic Convolutional Nueral Network implement. Here can use for 2D, 3D or 4D data sets.

        Arguments:
            n_classes(Integer): How many classes you want to classifier or use it for regression.
                Integer 2 is for binary problem, Integer >2 is for how many classes you have,
                Integert -1 is for regression problem.

            input_dim_1(Integer): First Dimension of your data sets. Noted, this is not batch-size dimension.

            input_dim_2(Integer): Second Dimension of your data sets.

            input_dim_3(Integer, optional): Third Dimension of your data sets. Noted, if data sets is just 2D,
                this parameter should not be given.

            is_2D(Boolean, optional): Whether or not data sets is just 2D.
                Default is True.

            is_3D(Boolean, optional): Whether or not data sets is just 3D.
                Default is False.

            n_conv_layers(Integer, optional): How many convolutional layers wanted to construct.

            conv_units(Integet, optional): How many units for each Convolutional layer.
                For better accuracy, give more units will be good.

            kernel_size(Integet, optional): How big is kernel wanted to slide on data sets.
                This kernel will be (n*n)

            stride(Integer, optional): How many steps make the kernel slide on data sets.

            padding(String, optional): Whether or not drop or fill the missing values.
                Default is 'SAME': returned result is same as before, if 'valid': missing value will be droped.

            use_global(Boolean, optional): Whether or not to use global average pooling.
                Default is False, meaning Flatten Conv result.

            use_pooling(Boolean, optional): Whether or not to use pooling after Convolutional layer.
                Default if False, meaning without Pooling layer.

            use_dropout(Boolean, optional): Whether or not to use Dropout after each layer.
                Default is True, recommended when model is overfitting.

            drop_ratio(Double, optional): How many ratios to drop units during training.
                Default is 0.5, if wanted for bigger penalty, set this bigger, may cause underfitting.

            use_batch(Boolean, optional): Whether or not to use BatchNormalization durng training.
                Default is True, this is great for improving model training accuracy

            activation(String, optional): Which activation function wanted to be used.
                Default is 'relu'

            use_dnn(Boolean, optional): Whether or not to use the Dense layer.
                Default is True

            n_dnn_layers(Integer, optional): How many dense layer wanted to be used.

            dnn_units(Integer, optional): How many units to be used for each layer.

            loss(String, optional): Which loss function wanted to be used during training as objective function.
                Default is None, it will be choosen according to different problems. For 'binary' problem, use
                'binary_crossentropy', for 'multi-class' problem, use 'categorical_crossentropy', for 'regression'
                problem, use 'mse'.

            metrics(String, optional): Which metric wanted to be used during training.

            optimizer(String, optional): Which optimizer to be used for optimizing.
    """
    def __init__(self, n_classes, input_dim_1, input_dim_2, input_dim_3=None, is_2D=True, is_3D=False,
                 n_conv_layers=3, conv_units=128, kernel_size=2, stride=1, padding='SAME', use_global=False, use_pooling=False,
                 use_dropout=True, drop_ratio=.5, use_batch=True, activation='relu', use_dnn=True, n_dnn_layers=1, dnn_units=256,
                 loss=None, metrics='accuracy', optimizer='rmsprop'):
        super().__init__()
        self.n_classes = n_classes
        self.is_2D = is_2D
        self.is_3D = is_3D
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.input_dim_3 = input_dim_3
        self.n_conv_layers = n_conv_layers
        self.conv_units = conv_units
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_global = use_global
        self.use_pooling = use_pooling
        self.use_dropout = use_dropout
        self.drop_ratio = drop_ratio
        self.use_batch = use_batch
        self.activation = activation
        self.use_dnn = use_dnn
        self.n_dnn_layers = n_dnn_layers
        self.dnn_units = dnn_units
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.model = self._init_model()

    # Basic private function to construct CNN kernel function.
    def _basic_cnn(self, layer=None, name_index=None):
        if self.dimstr == '2D':
            res = Conv1D(self.conv_units, self.kernel_size, self.stride,
                         padding=self.padding)(layer)
            if self.use_pooling:
                res = MaxPooling1D(name='maxpool_'+str(name_index))
        elif self.dimstr == '3D':
            res = Conv2D(self.conv_units, self.kernel_size, self.stride,
                         padding=self.padding, name='conv_' + str(name_index))(layer)
            if self.use_pooling:
                res = MaxPooling2D(name='maxpool_'+str(name_index))(res)

        if self.use_batch:
            res = BatchNormalization(name='batch_' + str(name_index))(res)
        res = Activation(self.activation, name='activation_' + str(name_index))(res)
        if self.use_dropout:
            res = Dropout(self.drop_ratio)(res)

        return res

    # this is used for building basic CNN model.
    def _init_model(self):
        # According to parameters to check data is '2D' or '3D'
        self.dimstr = check_dims(self.input_dim_3, self.is_2D, self.is_3D)

        if self.dimstr == '2D':
            inputs = Input(shape=(self.input_dim_1, self.input_dim_2), name='inputs')
        elif self.dimstr == '3D':
            inputs = Input(shape=(self.input_dim_1, self.input_dim_2, self.input_dim_3), name='inputs')

        # loop for convolution layers
        for i in range(self.n_conv_layers):
            if i == 0:
                res = self._basic_cnn(inputs, i)
            else:
                res = self._basic_cnn(res, i)

        # whether or not use global average pooling layer
        if self.use_global:
            if self.dimstr == '2D':
                res = GlobalAveragePooling1D(name='global_1')(res)
            elif self.dimstr == '3D':
                res = GlobalAveragePooling2D(name='global_1')(res)
        else:      # if not global average pooling or Flatten Conv result
            res = Flatten()(res)

        # whether or not use Dense layer
        if self.use_dnn:
            for _ in range(self.n_dnn_layers):
                res = Dense(self.dnn_units, name='dense_1')(res)
                if self.use_batch:
                    res = BatchNormalization(name='dense_batch_1')(res)
                res = Activation(self.activation)(res)
                if self.use_dropout:
                    res = Dropout(self.drop_ratio, name='dense_drop_1')(res)

        # this is method private function to check whether or not loss is not given, then use default loss
        def _check_loss(model, loss, metrics, optimizer):
            if loss is not None:
                model.compile(loss=loss, metrics=[metrics], optimizer=optimizer)
            return model

        if self.n_classes == 2:      # this is binary class problem.
            out = Dense(self.n_classes, activation='sigmoid')(res)
            model = Model(inputs, out)
            if self.loss is None:
                model.compile(loss='binary_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
            else:  _check_loss(model, self.loss, self.metrics, self.optimizer)
        elif self.n_classes >= 2:    # this is multiclass problem.
            out = Dense(self.n_classes, activation='softmax')(res)
            model = Model(inputs, out)
            if self.loss is None:
                model.compile(loss='categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
            else: _check_loss(model, self.loss, self.metrics, self.optimizer)
        elif self.n_classes == -1:     # this is regression problem
            out = Dense(1)(res)
            model = Model(inputs, out)
            if self.loss is None:
                model.compile(loss='mse', metrics=[self.metrics], optimizer=self.optimizer)
            else: _check_loss(model, self.loss, self.metrics, self.optimizer)
        else:
            raise AttributeError("Parameter 'n_classes' should be -1, 2 or up 2!")

        print('Model structure summary:')
        model.summary()

        return model

