from typing import List

import keras


def create_dense_network(input_layer: "keras.Layer",
                         batch_normalization: bool = True,
                         dropout: float = 0.3,
                         num_of_neurons: List[int] = [50, 50, 20]) -> "keras.Layer":
    """
    :param input_layer: Input layer
    :param dropout: Percentage of neurons to drop
    :param batch_normalization: Boolean, if to normalize batch
    :param num_of_neurons: List of with number of neurons in neural network
    :return: Layer
    """

    layer = input_layer

    for neuron in num_of_neurons:
        layer = dense_layer(layer, neuron, dropout, batch_normalization)

    return layer


def dense_layer(input_layer: "keras.Layer",
                num_of_neurons: int,
                dropout: float,
                batch_normalization: bool = True,
                activation: bool = True) -> "keras.Layer":
    """
    :param input_layer:
    :param num_of_neurons:
    :param dropout:
    :param batch_normalization:
    :param activation:
    :return:
    """

    layer = keras.layers.Dense(num_of_neurons)(input_layer)

    if batch_normalization:
        layer = keras.layers.BatchNormalization()(layer)

    if activation:
        layer = keras.layers.LeakyReLU()(layer)

    layer = keras.layers.Dropout(dropout)(layer)
    return layer
