import json
import os.path as osp
import random

import numpy as np
from sklearn.model_selection import train_test_split

from settings import DATA_DIR, MODEL_WEIGHTS_DIR
from source.models.model import CNNModel
from source.utils import calculate_weights


def training(parameters):
    X_train = np.load(osp.join(DATA_DIR, 'train.npy')) / 255.0
    X_test = np.load(osp.join(DATA_DIR, 'test.npy')) / 255.0
    y_train = np.load(osp.join(DATA_DIR, 'train_target.npy'))
    y_test = np.load(osp.join(DATA_DIR, 'test_target.npy'))

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    if not parameters['class_weight']:
        weight_class = calculate_weights(y_train)
        with open('./model_weights/weights.json', 'w') as file:
            json.dump(weight_class.tolist(), file)
    else:
        with open(osp.join(MODEL_WEIGHTS_DIR, 'weights.json'), 'r') as file:
            weight_class = json.load(file)

    CNN = CNNModel(**parameters)
    if parameters['train']:
        CNN.build_model()
        CNN.train(X_train, y_train, X_valid, y_valid, weight_class=weight_class)
    else:
        CNN.load_model(parameters['load_dir'])



random.seed(42)
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', help='Whether to train data', type=bool, default=True)
    parser.add_argument('-e', '--epochs', help='Path to input data', type=int, default=100)
    parser.add_argument('-sd', '--save_dir', help='Path where to save model', default=MODEL_WEIGHTS_DIR)
    parser.add_argument('-ld', '--load_dir', help='Path to saved model', default='546-0.135-0.957-0.297-0.945.hdf5')
    # parser.add_argument('-ld', '--load_dir', help='Path to saved model', default='546-0.135-0.957-0.297-0.945.hdf5')
    parser.add_argument('-cw', '--class_weight', help='Path to class weights', default='weights.json')

    args = parser.parse_args()
    train = args.train
    epochs = args.epochs
    save_dir = args.save_dir
    class_weight = args.class_weight
    load_dir = args.load_dir

    parameters = {
        'train': train,
        'epochs': epochs,
        'save_dir': save_dir,
        'load_dir': load_dir,
        'class_weight': class_weight
    }

    training(parameters)
