import numpy as np
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances


def compute_distance_matrix(data, metric='euclidean'):

    """
    Computes pairwise distances according to a specified metric
    ------------------------------------------------------------

    :param data: (pandas DataFrame) DataFrame with columns representing dimensions using which the distance is computed
    :param metric: (str) Type of metric to be used ('euclidean' is used by default)

    :return: (numpy array) Distance matrix
    """

    if metric == 'kl_divergence':
        # Replacing zeros with very small values to prevent KL divergence from going to infinity
        # (due to division by zero)
        data[data == 0] = 10**-10

        # Computing distance matrix
        distance_matrix = np.zeros((data.shape[0], data.shape[0]))
        for row_idx in range(distance_matrix.shape[0]):
            for col_idx in range(distance_matrix.shape[1]):
                distance_matrix[row_idx, col_idx] = entropy(data.iloc[row_idx], data.iloc[col_idx])

    else:
        distance_matrix = pairwise_distances(data, metric=metric)

    return distance_matrix










