import csv
import os.path as osp
from typing import List

import pandas as pd

from settings import DATA_DIR


def read_from_csv(csv_path: str) -> List:
    """
    Reading from csv file, better with csv library than with pandas, not all columns have labels
    """
    rows = []
    with open(csv_path) as csvfile:
        train_lines = csv.reader(csvfile)
        for row in train_lines:
            rows.append(row)

    return rows


def append_image_num(base_column_names: List, number: int) -> List:
    """
    Add cipher name to base column name
    """
    return [column_name + str(number) for column_name in base_column_names]


def fill_column_names(df: pd.DataFrame,
                      num_of_attributes_per_cipher: int = 5,
                      skip_first_columns: int = 1,
                      take_first_row_as_header: bool = True) -> pd.DataFrame:
    """
    Not all columns have labels. The goal of this function is to take dataframe, find maximum number of images
    and fill correct column labels. We will start with list of starting column names and filling None values
    """

    if take_first_row_as_header:
        df.columns = df.iloc[0]
        df = df[1:]

    base_column_names = ['x_', 'y_', 'w_', 'h_', 'label_']

    column_names = list(df.columns)
    for index in range(skip_first_columns, len(column_names), num_of_attributes_per_cipher):

        # if column index is None, fill next num_of_attributes_per_cipher
        if column_names[index] is None:
            image_number = index // num_of_attributes_per_cipher + 1
            new_column_names = append_image_num(base_column_names, image_number)
            column_names[index: index + num_of_attributes_per_cipher] = new_column_names

    df.columns = column_names
    return df


def transform_df_to_one_instance_per_row(df: pd.DataFrame, num_of_attributes_per_cipher: int = 5) -> pd.DataFrame:
    """
    Iterate through all rows, split each row containing multiple images to single row with one image containing
    image name, bounding box and class
    """
    data = []
    for line in df.iterrows():
        row_value = list(line[1].dropna())
        # from row create multiple lists where each list contains one image property, without filename
        cipher_properties = [row_value[i:i + num_of_attributes_per_cipher] for i in
                             range(1, len(row_value), num_of_attributes_per_cipher)]
        for image in cipher_properties:
            new_line = [row_value[0].split("/")[1]]
            new_line.extend(image)
            data.append(new_line)
    new_df = pd.DataFrame(columns=['filename', 'x', 'y', 'w', 'h', 'class'], data=data)
    return new_df


def convert_ten_to_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some classes have 10 as target, we will change this to 0
    """
    mapping = dict((str(i), i) for i in range(10))
    mapping["10"] = 0

    df['class'] = df['class'].map(mapping)
    return df


def all_in_one(input_path: str, output_path: str):
    """
    Join whole pipeline from reading annotations to preprocessing and saving new files which will be used later
    """
    annotations_path = osp.join(DATA_DIR, input_path)
    rows = read_from_csv(annotations_path)

    df = pd.DataFrame.from_records(rows)
    df = fill_column_names(df)
    df = transform_df_to_one_instance_per_row(df)
    df = convert_ten_to_zero(df)

    df.to_csv(osp.join(DATA_DIR, output_path), index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--input_path', help='Path to input data', default='train_annotations.csv')
    parser.add_argument('-op', '--output_path', help='Path to output data',
                        default='preprocessed_train_annotations.csv')

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    all_in_one(input_path, output_path)
