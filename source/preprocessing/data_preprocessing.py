import os
import os.path as osp
import random
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

from settings import TRAIN_DIR, DATA_DIR, IMAGE_WIDTH_THRESHOLD, IMAGE_HEIGHT_THRESHOLD, CIPHER_HEIGHT_THRESHOLD, \
    CIPHER_WIDTH_THRESHOLD, PREPROCESSED_TRAIN_DIR, TEST_DIR, PREPROCESSED_TEST_DIR

random.seed(42)


def check_image_size(image, width_threshold: int = IMAGE_WIDTH_THRESHOLD,
                     height_threshold: int = IMAGE_HEIGHT_THRESHOLD):
    """

    """
    return image.width >= width_threshold and image.height >= height_threshold


def cut(annotations_path, list_dir, save_dir, save_name):
    """
    Iterate through images. Drop all that are smaller than 80x40. Drop than all cipher that are smaller
    than 20x20. Crop ciphers from the original image and resize them to 20x20 (all need to have same size).
    """

    annotations: pd.DataFrame = pd.read_csv(osp.join(DATA_DIR, annotations_path))
    data = []
    classes = []

    images: List[str] = os.listdir(list_dir)
    for image_name in images:
        if image_name.endswith('.png'):
            image = Image.open(osp.join(list_dir, image_name))
            if check_image_size(image):
                df: pd.DataFrame = annotations.loc[annotations['filename'] == image_name]

                # iterate through all ciphers in image
                for index, row in enumerate(df.iterrows()):
                    filename, x, y, w, h, clas = row[1]
                    if w > CIPHER_WIDTH_THRESHOLD and h > CIPHER_HEIGHT_THRESHOLD:
                        # left, top, right, bottom
                        crop_image = image.crop((x, y, x + w, y + h))
                        resized_image = crop_image.resize((CIPHER_WIDTH_THRESHOLD, CIPHER_HEIGHT_THRESHOLD))

                        resized_image.save(osp.join(save_dir, filename.split('.')[0] + str(index) + '.png'))

                        data.append(np.array(resized_image, dtype=np.float))
                        classes.append(clas)

    np.save(osp.join(DATA_DIR, f'{save_name}.npy'), data)
    np.save(osp.join(DATA_DIR, f'{save_name}_target.npy'), classes)


if __name__ == '__main__':
    cut('preprocessed_train_annotations.csv', list_dir=TRAIN_DIR, save_dir=PREPROCESSED_TRAIN_DIR, save_name='train')
    cut('preprocessed_test_annotations.csv', list_dir=TEST_DIR, save_dir=PREPROCESSED_TEST_DIR, save_name='test')

    print()
