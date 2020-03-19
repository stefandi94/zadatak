import os.path as osp

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from settings import DATA_DIR, BASE_DIR
from source.mean_shift import MeanShift
from source.utils import transparent_cmap


def plot_heatmap(csv_path, image_path, bandwidth):
    df = pd.read_csv(csv_path, header=None, names=['time_stamp', 'x_coordinate', 'y_coordinate'], delimiter=';')
    image = Image.open(image_path)

    df['x_coordinate'] = df['x_coordinate'].map(lambda x: int(0.5 * x + 150))
    df['y_coordinate'] = df['y_coordinate'].map(lambda x: abs(image.height - int(0.55 * x + 320)))

    points = []
    w, h = image.size

    for i in range(len(df['x_coordinate'])):
        points.append([df['x_coordinate'][i], df['y_coordinate'][i]])

    new_points = [[w - point[0], point[1]] for point in points]
    x_coord = [point[0] for point in new_points]
    y_coord = [point[1] for point in new_points]

    ms = MeanShift(bandwidth=bandwidth,
                   centroid_threshold=9)
    ms.fit(image, points)
    predictions = ms.predict(new_points)
    n_clusters = ms.n_clusters

    colors = 'bgrcmyk' * (n_clusters // 7 + 1)
    x_centers = [x[0] for x in ms.cluster_centers]
    y_centers = [x[1] for x in ms.cluster_centers]

    plt.figure(figsize=(12, 15))
    plt.scatter(x_coord, y_coord, c=predictions, cmap=matplotlib.colors.ListedColormap(colors))
    plt.scatter(x_centers, y_centers, c='r', s=150, label='Centroids')
    plt.legend()
    plt.title(f'Mean shift clustering with bandwidth {bandwidth}')
    plt.savefig(osp.join(BASE_DIR, 'images', 'meanshift_clustering.png'))
    plt.show()

    mycmap = transparent_cmap(plt.cm.Reds)
    heatmap = ms.heat_map(image, points)
    img = np.flip(np.fliplr(heatmap))

    y, x = np.mgrid[0:h, 0:w]

    plt.clf()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    cb = ax.contourf(x, y, img, 50, cmap=mycmap)
    plt.colorbar(cb)
    plt.title(f'Heatmap with bandwidth {bandwidth}')
    plt.savefig(osp.join(BASE_DIR, 'images', 'heatmap.png'))
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--csv_path', help='Path to csv file', default=osp.join(DATA_DIR, 'Heineken.csv'))
    parser.add_argument('-ip', '--image_path', help='Path to image file', default=osp.join(DATA_DIR, 'Heineken.jpg'))
    parser.add_argument('-bw', '--bandwidth', type=int, default=35)

    args = parser.parse_args()
    csv_path = args.csv_path
    image_path = args.image_path
    bandwidth = args.bandwidth

    plot_heatmap(csv_path, image_path, bandwidth=bandwidth)
