from typing import List

import numpy as np
from sklearn.utils import class_weight


def calculate_weights(y_train: np.ndarray or List[int]) -> np.ndarray:
    """
    Given class labels of training sample, calculates frequency of each class and return inverted proportion, to
    balance dataset.
    :param y_train: training labels
    :return: weight class for each label
    """

    weight_class = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    return weight_class


def transparent_cmap(cmap, N=255):
    """
    Copy colormap and set alpha values
    """
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
    return mycmap