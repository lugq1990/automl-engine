# -*- coding:utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Conv2D, Conv1D, \
     GlobalAveragePooling1D, GlobalAveragePooling2D, concatenate, Flatten, add

from auto_ml.nueral_networks.basic_net import BaseNet


class WideDeepNet(BaseNet):
    """This class implements Wide & Deep deep learning models.

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

                n_wide_layers(Integer, optional): How many wide layers to be used.

                first_conv(Boolean, optional): Whether to use Convolutional layer at before layers.

                n_first_convs(Integer, optional): How many Convolutional layers to be used at before layers.

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
    def __init__(self, n_classes, input_dim_1, input_dim_2, input_dim_3=None, is_2D=True, is_3D=False,
                 n_wide_layers=3, first_conv=True, n_first_convs=1, layer_concat=False, kernel_size=2, stridess=1, padding='SAME',
                 use_global=False, use_pooling=False, use_dnn=True, n_dnn_layers=1, conv_units=64, dnn_units=128, use_dropout=True,
                 drop_ratio=.5, use_batch=True, activation='relu', loss=None, optimizer='rmsprop', metrics='accuracy', silence=False):
        super().__init__()
        self.n_classes = n_classes
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.input_dim_3 = input_dim_3
        self.is_2D = is_2D
        self.is_3D = is_3D
        self.n_classes = n_classes
        self.n_wide_layers = n_wide_layers
        self.first_conv = first_conv
        self.n_first_convs = n_first_convs
        self.layer_concat = layer_concat
        self.use_global = use_global
        self.use_dnn = use_dnn
        self.use_pooling = use_pooling
        self.n_dnn_layers = n_dnn_layers
        self.conv_units = conv_units
        self.kernel_size = kernel_size
        self.strides = stridess
        self.padding = padding
        self.dnn_units = dnn_units
        self.use_dropout = use_dropout
        self.drop_ratio = drop_ratio
        self.use_batch = use_batch
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.silence = silence
        self.model = self._init_model()

    # This private function is to construct wide deep block.
    def _wide_conv(self, conv, layers):
        wide_1 = conv(self.conv_units, 1, padding=self.padding)(layers)

        wide_2 = conv(self.conv_units, 1, padding=self.padding)(layers)
        wide_2 = conv(self.conv_units, 3, padding=self.padding)(wide_2)

        wide_3 = conv(self.conv_units, 1, padding=self.padding)(layers)
        wide_3 = conv(self.conv_units, 5, padding=self.padding)(wide_3)

        wide_4 = conv(self.conv_units, 3, padding=self.padding)(layers)
        wide_4 = conv(self.conv_units, 1, padding=self.padding)(wide_4)

        # whether or not to use conconcate to merge layer or just add them up.
        if self.layer_concat:
            return concatenate([wide_1, wide_2, wide_3, wide_4], axis=0)
        else:
            return add([wide_1, wide_2, wide_3, wide_4])

    # This is wide and deep block construction function.
    # GoogleNet Inception block
    def _wide_deep_block(self, layers):
        if self._check_dims(self.input_dim_3, self.is_2D, self.is_3D) == '2D':
            res = self._wide_conv(Conv1D, layers)
        elif self._check_dims(self.input_dim_3, self.is_2D, self.is_3D) == '3D':
            res = self._wide_conv(Conv2D, layers)

        return res

    # Main model sturcture construction function
    def _init_model(self):
        # Check given parameters to judge data is '2D' or '3D'
        self.dimstr = self._check_dims(self.input_dim_3, self.is_2D, self.is_3D)
        
        if self.dimstr == '2D':
            inputs = Input(shape=(self.input_dim_1, self.input_dim_2))
        elif self.dimstr == '3D':
            inputs = Input(shape=(self.input_dim_1, self.input_dim_2, self.input_dim_3))

        # Whether first use Convolutional layer, can be choosen.
        if self.first_conv:
            for _ in range(self.n_first_convs):
                if self.dimstr == '2D':
                    res_conv = Conv1D(self.conv_units, self.kernel_size, self.strides, padding=self.padding)(inputs)
                elif self.dimstr == '3D':
                    res_conv = Conv2D(self.conv_units, self.kernel_size, self.strides, padding=self.padding)(inputs)
        else:
            res_conv = inputs

        # Here is Wide & Deep model Block
        for i in range(self.n_wide_layers):
            if i == 0:
                res = self._wide_deep_block(res_conv)
            else:
                res = self._wide_deep_block(res)

        # Whether to use global avarega pooling or just Flatten concolutional result
        if self.use_global:
            if self.dimstr == '2D':
                res = GlobalAveragePooling1D()(res)
            elif self.dimstr == '3D':
                res = GlobalAveragePooling2D()(res)
        else:
            res = Flatten()(res)

        # Whether to use Dense layers
        if self.use_dnn:
            for _ in range(self.n_dnn_layers):
                res = Dense(self.dnn_units)(res)
                if self.use_batch:
                    res = BatchNormalization()(res)
                res = Activation(self.activation)(res)
                if self.use_dropout:
                    res = Dropout(self.drop_ratio)(res)

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
                model.compile(loss='sparse_categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
            else:
                _check_loss(model, self.loss, self.metrics, self.optimizer)
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

    @staticmethod
    def _check_dims(dims, is_2d, is_3d):
        if is_2d and dims is None:
            return '2D'
        elif is_3d or dims is not None:
            return '3D'


if __name__ == '__main__':
    import numpy as np

    data = np.random.randn(100, 10, 10)
    label = np.array([np.random.randint(0, 3) for _ in range(100)]).reshape(-1, 1)
    wide_deep_model = WideDeepNet(3, 10, 10)

    his = wide_deep_model.fit(data, label)

    score = wide_deep_model.score(data, label)

    model_name = wide_deep_model.__class__.__name__ + '-' + str(score)

    wide_deep_model.save(model_name)
