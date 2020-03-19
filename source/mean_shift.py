import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


class MeanShift:
    def __init__(self, bandwidth: float, centroid_threshold: float = 10):
        """
        MeanShift algorithm for clustering.
        """
        self.bandwidth = bandwidth
        self.radius = 6 * bandwidth + 1
        self.filter = self.gauss_kernel()

        self.point_to_center = dict()
        self.point_to_temp_cent = dict()
        self.label_to_centroid = dict()
        self.temp_cent_to_center = dict()
        self.centroid_to_label = dict()

        self.kde_matrix = None

        self.n_clusters = 0
        self.cluster_centers = []
        self.temp_cluster_centers = []
        self.centroid_threshold = centroid_threshold

    @staticmethod
    def preprocessing(image, points):
        X = np.zeros((image.width, image.height))

        for point in points:
            X[point[0], point[1]] = 1
        X = np.fliplr(X.T)
        return X

    def find_weighted_average(self, matrix):
        """
        Given 3D points, return coordinates of center of the input matrix and value.
        """
        i, j = ndimage.measurements.center_of_mass(matrix)
        value = matrix.sum() / (self.N * self.bandwidth ** 2)
        return i, j, value

    def fit(self, image, points):
        """
        Fitting meanshift algorithm, where input is matrix with shape (x, y, 1). First pad this matrix with given radius.
        Iterate through all this points, multiplicate neighbourhood with gaussian kernel, elementwise.
        Find center of the mass of new matrix, this will be current mean shift. Repeat this process until convergence
        (until mode is found, algorithm converges).
        """
        X = self.preprocessing(image, points)
        self.N = X.sum()
        self.kde_matrix = np.zeros((image.width, image.height))
        new_X = np.pad(X, self.radius // 2)
        for i in range(self.radius // 2, new_X.shape[0] - self.radius // 2):
            for j in range(self.radius // 2, new_X.shape[1] - self.radius // 2):
                starting_position = [i, j]

                # skip zero values points
                if new_X[i, j] == 0:
                    continue

                mode_coordinates = starting_position.copy()
                mode_found = False
                mode = 0

                while not mode_found:
                    # elementwise multiplication of point around radius with (gaussian) filter
                    probability_matrix = np.multiply(
                        new_X[starting_position[0] - self.radius // 2: starting_position[0] + self.radius // 2 + 1,
                        starting_position[1] - self.radius // 2: starting_position[1] + self.radius // 2 + 1],
                        self.filter)

                    # find center of the mass of this probability matrix and moving current mode and staring position
                    mode_i, mode_j, value = self.find_weighted_average(probability_matrix)
                    self.kde_matrix[j - self.radius // 2, i - self.radius // 2] = value

                    starting_position = [starting_position[0] + int(mode_i) - self.radius // 2,
                                         starting_position[1] + int(mode_j) - self.radius // 2]

                    mode_coordinates = [starting_position[0] + mode_i - self.radius // 2,
                                        starting_position[1] + mode_j - self.radius // 2]

                    if value > mode:
                        mode = value
                    else:
                        mode_found = True

                self.point_to_temp_cent[i - self.radius // 2, j - self.radius // 2] = mode_coordinates
                if mode_coordinates not in self.temp_cluster_centers:
                    self.temp_cluster_centers.append(mode_coordinates)

        self.prune_unnecessary_centroids()
        self.n_clusters = len(self.cluster_centers)

        # x_centers = [x[1] - radius // 2 for x in ms.cluster_centers]
        # y_centers = [x[0] - radius // 2 for x in ms.cluster_centers]
        self.cluster_centers = [[center[1] - self.radius // 2, center[0] - self.radius // 2] for center in
                                self.cluster_centers]
        self.label_to_centroid = dict((i, self.cluster_centers[i]) for i in range(len(self.cluster_centers)))
        self.centroid_to_label = dict((tuple(value), key) for key, value in self.label_to_centroid.items())

    def predict(self, X):
        predictions = []
        for point in X:
            closest_center = self.find_nearest(self.cluster_centers, point)
            predictions.append(self.centroid_to_label[tuple(closest_center)])
        return predictions

    @staticmethod
    def find_nearest(centers, point):
        centers = np.asarray(centers)
        # for center in centers:

        idx = np.array((centers - point) ** 2).sum(axis=1).argmin()
        return centers[idx]

    def find_minimum_distance(self, center):
        """
        Given center, iterate through all other centroids and return list of centers within threshold
        """

        close_centers = []
        for other_center in self.temp_cluster_centers:
            if (np.array(other_center) != np.array(center)).all():
                distance = np.linalg.norm([np.array(center) - np.array(other_center)])
                if distance < self.centroid_threshold:
                    close_centers.append(other_center)
        return close_centers

    def prune_unnecessary_centroids(self):
        """
        Iterate through all temporary cluster centers. If there are other cluster centers within given threshold,
        take mean, delete all centers in that radius, add new cluster center to list of temporary centroids.
        If there are no other centroids in the neighbourhood, add that center to cluster centers and remove it
        from the temporary list. Stop when temporary list is empty
        """
        while len(self.temp_cluster_centers):
            cluster_center = self.temp_cluster_centers[0]
            close_centers = self.find_minimum_distance(cluster_center)

            # check if there are other centroids
            if close_centers:
                close_centers.append(cluster_center)

                new_center = list(np.mean(close_centers, axis=0))
                for old_center in close_centers:
                    self.temp_cluster_centers.remove(old_center)
                    self.temp_cent_to_center[old_center[0], old_center[1]] = cluster_center

                if new_center not in self.cluster_centers:
                    self.temp_cluster_centers.append(list(new_center))
                    self.cluster_centers.append(list(new_center))
            # there are no other centroids, this is "real centroid"
            else:
                if cluster_center not in self.cluster_centers:
                    self.cluster_centers.append(cluster_center)
                self.temp_cluster_centers.remove(cluster_center)
                self.temp_cent_to_center[cluster_center[0], cluster_center[1]] = cluster_center

    def gauss_function(self, x, y):
        # (1 / 2*pi*h) * exp(-1/2 *((x/h)**2 + (y/h)**2)
        return (1 / (2 * np.pi * self.bandwidth ** 2)) * \
               np.exp(-0.5 * ((x / self.bandwidth) ** 2 + (y / self.bandwidth) ** 2))

    def gauss_kernel(self):
        """
        Create a 2D Gaussian kernel array.
        """
        # create nxn zeros
        gauss_filter = np.zeros((self.radius, self.radius))
        for i in range(gauss_filter.shape[0]):
            for j in range(gauss_filter.shape[1]):
                gauss_filter[i, j] = self.gauss_function(i - self.radius // 2, j - self.radius // 2)
        return gauss_filter

    def heat_map(self, image, points):
        X = self.preprocessing(image, points)
        heatmap = gaussian_filter(X, self.bandwidth)
        return heatmap
