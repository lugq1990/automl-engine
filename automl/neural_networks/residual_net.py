# -*- coding:utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dense, Dropout, Activation, \
    GlobalAveragePooling1D, Flatten, MaxPooling1D, MaxPooling2D, Conv2D, GlobalAveragePooling2D, add, concatenate

from .basic_net import BaseNet


class ResidualNet(BaseNet):
    """This class implements basic residualNet, also with DenseNet.
    
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
                
            basic_residual(Boolean, optional): Whether to use basic ResidualNet.
                Default is True, if set this param to False, then will use DenseNet.

            n_res_layers(Integer, optional): How many residual layers wanted to construct.

            conv_units(Integet, optional): How many units for each Convolutional layer.
                For better accuracy, give more units will be good.

            kernel_size(Integet, optional): How big is kernel wanted to slide on data sets.
                This kernel will be (n*n)

            strides(Integer, optional): How many steps make the kernel slide on data sets.

            padding(String, optional): Whether or not drop or fill the missing values.
                Default is 'SAME': returned result is same as before, if 'valid': missing value will be droped.

            use_global(Boolean, optional): Whether or not to use global average pooling.
                Default is False, meaning Flatten Conv result.

            use_pooling(Boolean, optional): Whether or not to use pooling after Convolutional layer.
                Default if False, meaning without Pooling layer.

            use_concat(Boolean, optional): Whether to use 'concat' or 'add' to merge layers.
                Default is False use 'add' to merge layers.

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

            silence(Boolean, optional): Whether to print some useful information to console.
                Default is False, means print something to console.
    
    """
    def __init__(self, n_classes, input_dim_1, input_dim_2, input_dim_3=None,  is_2D=True, is_3D=False,
                 basic_residual=False, n_res_layers=4, kernel_size=2, stridess=1, padding='SAME',
                 use_global=False, use_dense=True, use_pooling=False, use_concat=False,
                 n_dnn_layers=1, conv_units=64, dnn_units=128, use_dropout=True,
                 drop_ratio=.5, use_batch=True, loss=None, optimizer='rmsprop', metrics='accuracy', silence=False):
        super(ResidualNet, self).__init__()
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.input_dim_3 = input_dim_3
        self.is_2D = is_2D
        self.is_3D = is_3D
        self.n_classes = n_classes
        self.basic_residual = basic_residual
        self.n_res_layers = n_res_layers
        self.use_global = use_global
        self.use_dense = use_dense
        self.use_pooling = use_pooling
        self.use_concat = use_concat
        self.n_dnn_layers = n_dnn_layers
        self.conv_units = conv_units
        self.kernel_size = kernel_size
        self.strides = stridess
        self.padding = padding
        self.dnn_units = dnn_units
        self.use_dropout = use_dropout
        self.drop_ratio = drop_ratio
        self.use_batch = use_batch
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.silence = silence
        self.model = self._init_model()

    # Residual block including for using basic ResidualNet or use DenseNet.
    # This including 2 convolutional layers in each block.
    def _res_block(self, layers, added_layers):
        # Is 2D or 3D data, also with whether or not to use pooling layer.
        if self.dimstr == '2D':
            res = Conv1D(self.conv_units, self.kernel_size, self.strides, padding=self.padding)(layers)
            if self.use_pooling:
                res = MaxPooling1D()(res)
        elif self.dimstr == '3D':
            res = Conv2D(self.conv_units, self.kernel_size, self.strides, padding=self.padding)(layers)
            if self.use_pooling:
                res = MaxPooling2D()(res)

        if self.use_batch:
            res = BatchNormalization()(res)
        res = Activation('relu')(res)
        if self.use_dropout:
            res = Dropout(self.drop_ratio)(res)

        # Is 2D or 3D data
        if self.dimstr == '2D':
            res = Conv1D(self.input_dim_2, self.kernel_size, self.strides, padding=self.padding)(res)
        elif self.dimstr == '3D':
            res = Conv2D(self.input_dim_2, self.kernel_size, self.strides, padding=self.padding)(res)

        if self.use_batch:
            res = BatchNormalization()(res)
        res = Activation('relu')(res)
        if self.use_dropout:
            res = Dropout(self.drop_ratio)(res)

        # This is basic ResidualNet.
        if self.basic_residual:
            if self.use_concat:
                return concatenate([res, layers])
            else:
                return add([res, layers])
        # This is DenseNet.
        else:
            if self.use_concat:
                return concatenate([res, added_layers])
            else:
                return add([res, added_layers])

    # This will build DenseNet or ResidualNet structure, model is already compiled.
    def _init_model(self):
        # Check the input parameters for getting data shape is '2D' or '3D'
        self.dimstr = check_dims(self.input_dim_3, self.is_2D, self.is_3D)
        inputs = Input(shape=(self.input_dim_1, self.input_dim_2))

        # construct residual block chain.
        for i in range(self.n_res_layers):
            if i == 0:
                res = self._res_block(inputs, inputs)
            else:
                res = self._res_block(res, inputs)

        # using flatten or global average pooling to process Convolution result
        if not self.use_global:
            res = Flatten()(res)
        else:
            if self.dimstr == '2D':
                res = GlobalAveragePooling1D()(res)
            elif self.dimstr == '3D':
                res = GlobalAveragePooling2D()(res)

        # whether or not use dense net, also with how many layers to use
        if self.use_dense:
            for _ in range(self.n_dnn_layers):
                res = Dense(self.dnn_units)(res)
                if self.use_batch: res = BatchNormalization()(res)
                res = Activation('relu')(res)
                if self.use_dropout: res = Dropout(self.drop_ratio)(res)

        # check parameter 'loss' is given or not, if not, use default loss for model training.
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
