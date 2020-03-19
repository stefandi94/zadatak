import os
import os.path as osp

BASE_DIR = osp.abspath(osp.dirname(__file__))

DATA_DIR = osp.join(BASE_DIR, 'data')
MODEL_WEIGHTS_DIR = osp.join(BASE_DIR, 'model_weights')
TRAIN_DIR = osp.join(DATA_DIR, 'train')
TEST_DIR = osp.join(DATA_DIR, 'test')

PREPROCESSED_TRAIN_DIR = osp.join(DATA_DIR, 'preprocessed_train')
PREPROCESSED_TEST_DIR = osp.join(DATA_DIR, 'preprocessed_test')

if not osp.exists(PREPROCESSED_TRAIN_DIR):
    os.makedirs(PREPROCESSED_TRAIN_DIR)

if not osp.exists(PREPROCESSED_TEST_DIR):
    os.makedirs(PREPROCESSED_TEST_DIR)

NUM_OF_CLASSES = 10

IMAGE_WIDTH_THRESHOLD = 80
IMAGE_HEIGHT_THRESHOLD = 40

CIPHER_WIDTH_THRESHOLD = 20
CIPHER_HEIGHT_THRESHOLD = 20
