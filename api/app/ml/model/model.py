import os
import os.path as osp
from contextlib import redirect_stdout
from typing import List, Tuple

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam

from api.settings import NUM_OF_CLASSES
from app.ml.model.cnn_layers import create_cnn_network
from app.ml.model.dense_layers import create_dense_network
from sklearn.utils import class_weight


class CNNModel:
    allowed_kwargs = ['epochs', 'batch_size', 'num_classes', 'learning_rate', 'optimizer', 'save_dir', 'load_dir']
    convolution_filters = [32, 64, 128, 128]

    def __init__(self,
                 epochs: int = 100,
                 batch_size: int = 64,
                 num_classes: int = NUM_OF_CLASSES,
                 **kwargs) -> None:

        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = [20, 20, 3]

        self.save_dir = None
        self.load_dir = None

        self.optimizer = None
        self.learning_rate = None

        for k in kwargs.keys():
            if k in self.allowed_kwargs:
                self.__setattr__(k, kwargs[k])

    def build_model(self) -> None:
        inputs = keras.layers.Input(self.input_shape)

        layer = create_cnn_network(inputs, self.convolution_filters, kernels=(3, 3),
                                   dropout=0.2, batch_normalization=True)

        layer = keras.layers.LeakyReLU()(layer)
        layer = create_dense_network(layer, num_of_neurons=[200, 100])

        layer = keras.layers.Dropout(0.3)(layer)
        output = keras.layers.Dense(self.num_classes, activation='softmax')(layer)

        model = keras.Model(inputs, output)
        self.model = model

    def train(self,
              X_train: np.ndarray or List[np.ndarray],
              y_train: np.ndarray or List[np.ndarray],
              X_valid,
              y_valid,
              weight_class: np.ndarray = None):
        """
        Train model given parameters
        :param X_train: train data
        :param y_train: train classes
        :param weight_class: weights for
        :return:
        """

        self.build_model()

        # Given path, creates folders to that path
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f'Created path for model at: {self.save_dir}')

        # If load path is given, load model from that path
        if self.load_dir is not None:
            self.load_model(self.load_dir)
            print(f'Model is loaded from {self.load_dir}')

        self.model.compile(loss=['sparse_categorical_crossentropy'],
                           optimizer=Adam(self.learning_rate, clipnorm=1, clipvalue=0.5),
                           metrics=['acc'])
        # metrics=['acc', self.precision, self.recall, self.f1])

        weights_name = "{epoch}-{loss:.3f}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}.hdf5"
        # weights_name = "{epoch}-{dense_1_loss:.3f}-{dense_1_acc:.3f}-{val_dense_1_loss:.3f}-{val_dense_1_acc:.3f}.hdf5"

        # print(self.model.summary())
        with open(osp.join(self.save_dir, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()
        # plot_model(self.model, to_file=osp.join(self.save_dir, 'model.png'), show_shapes=True, show_layer_names=True)

        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, weights_name),
                                     monitor='val_acc',
                                     verbose=1,
                                     save_weights_only=False,
                                     save_best_only=True,
                                     mode='max')

        csv_logger = CSVLogger(osp.join(self.save_dir, "model_history_log.csv"), append=True)
        callbacks_list = [checkpoint, csv_logger]

        history = self.model.fit(X_train, y_train,
                                 validation_data=(X_valid, y_valid),
                                 verbose=1,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 callbacks=callbacks_list,
                                 shuffle=True,
                                 class_weight=weight_class)

        return history

    def predict(self, X_test: List[np.ndarray] or np.ndarray) -> List[Tuple[int, float]]:
        """Return prediction for given data."""
        # self.load_model(self.load_dir)
        # print(f'Model is loaded from {self.load_dir}')

        all_predictions = self.model.predict(X_test)
        predicted_class = np.argmax(all_predictions, axis=1)
        confidence = np.array([all_predictions[vec_num][idx] for vec_num, idx in enumerate(predicted_class)])
        return list(zip(predicted_class, confidence))

    def load_model(self, path: str) -> None:
        """Load model from given path."""

        self.model = keras.models.load_model(path)
        # self.model = keras.models.load_model(path, custom_objects={'recall': self.recall,
        #                                                             'precision': self.precision,
        #                                                             'f1': self.f1})

    def save_model(self, path: str) -> None:
        """Save model on given path."""

        try:
            os.makedirs(osp.dirname(path), exist_ok=True)
            self.model.save(path)
        except Exception as e:
            print(e)
            print("Couldn't save model on path {}!".format(path))

    # @staticmethod
    # def recall(y_true, y_pred):
    #     """Recall metric.
    #
    #     Only computes a batch-wise average of recall.
    #
    #     Computes the recall, a metric for multi-label classification of
    #     how many relevant items are selected.
    #     """
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 0)))
    #     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #     recall = true_positives / (possible_positives + K.epsilon())
    #     return recall
    #
    # @staticmethod
    # def precision(y_true, y_pred):
    #     """Precision metric.
    #
    #     Only computes a batch-wise average of precision.
    #
    #     Computes the precision, a metric for multi-label classification of
    #     how many selected items are relevant.
    #     """
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #     precision = true_positives / (predicted_positives + K.epsilon())
    #     return precision

    # def f1(self, y_true, y_pred):
    #     precision = self.precision(y_true, y_pred)
    #     recall = self.recall(y_true, y_pred)
    #     return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def __str__(self):
        return 'CNN'


def calculate_weights(y_train: np.ndarray or List[int]) -> np.ndarray:
    """
    Given class labels of training sample, calculates frequency of each class and return inverted proportion, to
    balance dataset.
    :param y_train: training labels
    :return: weight class for each label
    """

    weight_class = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    return weight_class