# -*- coding:utf-8 -*-
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model

from auto_ml.nueral_networks.basic_net import BaseNet


class DnnNet(BaseNet):
    """This is basic DNN networks.
        Arguments:
            n_classes(Integer): How many classes you want to classifier or use it for regression.
                Integer 2 is for binary problem, Integer >2 is for how many classes you have,
                Integert -1 is for regression problem.

            n_dims(Integer): How many dimensions of training data.

            n_layers(Integer, optional): How many dense layers wanted to be used.

            n_units(Integer, optional): How many units to be used for each dense layer.

            use_dropout(Boolean, optional): Whether or not to use Dropout after each layer.
                Default is True, recommended when model is overfitting.

            drop_ratio(Double, optional): How many ratios to drop units during training.
                Default is 0.5, if wanted for bigger penalty, set this bigger, may cause underfitting.

            use_batch(Boolean, optional): Whether or not to use BatchNormalization durng training.
                Default is True, this is great for improving model training accuracy

            loss(String, optional): Which loss function wanted to be used during training as objective function.
                Default is None, it will be choosen according to different problems. For 'binary' problem, use
                'binary_crossentropy', for 'multi-class' problem, use 'categorical_crossentropy', for 'regression'
                problem, use 'mse'.

            metrics(String, optional): Which metric wanted to be used during training.

            optimizer(String, optional): Which optimizer to be used for optimizing.

            silence(Boolean, optional): Whether to print some useful information to console.
                Default is False, means print something to console.
    """
    def __init__(self, n_classes, n_dims, n_layers=3, n_units=64, use_dropout=True, drop_ratio=.5,
                 use_batch=True, loss=None, metrics='accuracy', optimizer='adam', silence=False):
        super().__init__()
        self.n_classes = n_classes
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.n_units = n_units
        self.use_dropout = use_dropout
        self.drop_ratio = drop_ratio
        self.use_batch = use_batch
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.silence = silence
        self.model = self._init_model()

    # this is dense block function.
    def _dense_block(self, layers):
        res = Dense(self.n_units)(layers)
        if self.use_batch:
            res = BatchNormalization()(res)
        res = Activation('relu')(res)
        if self.use_dropout:
            res = Dropout(self.drop_ratio)(res)
        return res

    def _init_model(self):
        inputs = Input(shape=(self.n_dims, ))
        for i in range(self.n_layers):
            if i == 0:
                res = self._dense_block(inputs)
            else:
                res = self._dense_block(res)

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
            else:
                _check_loss(model, self.loss, self.metrics, self.optimizer)
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


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)

    dnn = DnnNet(3, 4, n_layers=1, n_units=256)
    dnn.fit(x, y)

    print(dnn.score(x, y))
    print("Sample prob:", dnn.predict_proba(x)[:5])
    print("Sample pred:", dnn.predict(x)[:5])

