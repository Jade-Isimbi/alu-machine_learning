#!/usr/bin/env python3

"""
This module contains a function that
calculates total intra-cluster variance for a dataset
"""

import numpy as np


def variance(X, C):
    """
    calculates total intra-cluster variance for a dataset

    X: numpy.ndarray (n, d) containing the dataset
    C: numpy.ndarray (k, d) containing the centroid means for each cluster

    return:
        - var: total variance, or None on failure
    """
    if type(X) is not np.ndarray or type(C) is not np.ndarray:
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    n, d = X.shape
    k, d_c = C.shape
    if d != d_c or n == 0 or k == 0:
        return None
    # Squared distances: (k, n) - for each point, squared dist to each centroid
    sq_dists = np.sum((X[np.newaxis, :, :] - C[:, np.newaxis, :]) ** 2, axis=-1)
    # Min squared distance per point (nearest centroid), then sum
    var = np.sum(np.min(sq_dists, axis=0))
    return var
