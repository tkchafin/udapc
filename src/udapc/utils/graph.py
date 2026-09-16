import numpy as np
from sklearn.neighbors import NearestNeighbors

def local_affinity(X, n_neighbors=7):
    """
    Build a symmetric local affinity matrix using k-nearest neighbors.

    Args:
        X (np.ndarray): Input data (n_samples x n_features)
        n_neighbors (int): Number of neighbors

    Returns:
        np.ndarray: Affinity matrix A (n_samples x n_samples)
    """
    n = X.shape[0]
    A = np.zeros((n, n))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    for i in range(n):
        _, indices = nbrs.kneighbors([X[i]])
        for j in indices[0]:
            A[i, j] = 1
            A[j, i] = 1  # make symmetric
    return A
