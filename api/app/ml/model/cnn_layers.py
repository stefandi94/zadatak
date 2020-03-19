from typing import List, Tuple

# import tensorflow as tf
import keras

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0)


def create_cnn_layer(input_layer: "keras.Layer",
                     num_filter: int,
                     kernel_size: int or Tuple[int, int] = (3, 3),
                     dropout: float = 0.0,
                     batch_normalization: bool = False,
                     kernel_init: str or "keras.Initializer" = kernel_init,
                     bias_init: str or "keras.Initializer" = bias_init,
                     strides: int or Tuple[int, int] = (1, 1),
                     kernel_regularizer=None,
                     activation: bool = True) -> "keras.Layer":
    """
    Given input layer and number of filters, do 2D convolution
    :param input_layer: Input layer
    :param num_filter: Number of feature maps
    :param batch_normalization
    :param dropout:
    :param kernel_init:
    :param bias_init:
    :param kernel_size:
    :param strides
    :param kernel_regularizer
    :param activation
    :return: Layer
    """

    layer = keras.layers.Conv2D(num_filter,
                                strides=strides,
                                kernel_size=kernel_size,
                                padding='same',
                                kernel_initializer=kernel_init,
                                bias_initializer=bias_init,
                                kernel_regularizer=kernel_regularizer)(input_layer)

    if batch_normalization:
        layer = keras.layers.BatchNormalization()(layer)
    if activation:
        layer = keras.layers.LeakyReLU()(layer)

    layer = keras.layers.Dropout(dropout)(layer)

    return layer


def create_cnn_network(input_layer: "keras.Layer",
                       num_of_filters: List[int],
                       dropout: float = 0,
                       batch_normalization: bool = False,
                       kernels: List[int] or List[Tuple] = None) -> "keras.Layer":
    """
    Given input layer and number of filters, creates network
    :param input_layer:
    :param num_of_filters:
    :param kernels:
    :param dropout
    :param batch_normalization
    :return:
    """
    if kernels is None:
        kernels = (3, 3)

    layer = input_layer
    for index, filter in enumerate(num_of_filters):
        layer = create_cnn_layer(input_layer=layer,
                                 num_filter=filter,
                                 kernel_size=kernels,
                                 dropout=dropout,
                                 batch_normalization=batch_normalization)

    layer = keras.layers.GlobalAvgPool2D()(layer)
    return layer
