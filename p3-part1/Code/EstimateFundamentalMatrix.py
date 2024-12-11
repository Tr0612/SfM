import numpy as np
from numpy import linalg as LA
import math
import sys

def EstimateFundamentalMatrix(x1DF, x2DF):
    """
    Estimates the Fundamental matrix (F) using the 8-point algorithm with normalization.

    Args:
        x1DF (ndarray): Source image points (Nx2 array of [x, y]).
        x2DF (ndarray): Target image points (Nx2 array of [x, y]).

    Returns:
        F (ndarray): The estimated 3x3 Fundamental matrix.
    """
    def normalize_points(points):
        """Normalize points to have zero mean and mean distance sqrt(2)."""
        centroid = np.mean(points, axis=0)
        translated = points - centroid
        mean_dist = np.mean(np.linalg.norm(translated, axis=1))
        scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        normalized = (T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
        return normalized[:, :2], T

    # Ensure inputs are numpy arrays
    x1 = np.array(x1DF)
    x2 = np.array(x2DF)

    # Normalize points
    x1_norm, T1 = normalize_points(x1)
    x2_norm, T2 = normalize_points(x2)

    # Construct matrix A
    A = np.zeros((len(x1_norm), 9))
    for i in range(len(x1_norm)):
        x1, y1 = x1_norm[i]
        x2, y2 = x2_norm[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Solve A*f = 0 using SVD
    _, _, Vt = LA.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = LA.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    # Denormalize
    F = T2.T @ F @ T1

    # Scale F so that F[2, 2] = 1 (optional, for consistency)
    if F[-1, -1] != 0:
        F = F / F[-1, -1]

    return F
