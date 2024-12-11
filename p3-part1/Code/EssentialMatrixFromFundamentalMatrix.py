import numpy as np


# Function to compute the Essential Matrix from the Fundamental Matrix
def EssentialMatrixFromFundamentalMatrix(F, K):
    """
    EssentialMatrixFromFundamentalMatrix: Computes the Essential Matrix (E) from the Fundamental Matrix (F)
    and the camera intrinsic matrix (K).
    
    Parameters:
    - F: Fundamental matrix (3x3), relating corresponding points between two views in normalized image coordinates.
    - K: Intrinsic camera matrix (3x3), containing the intrinsic parameters of the camera.
    
    Returns:
    - new_E: Corrected Essential matrix (3x3) that enforces the constraints necessary for a valid Essential matrix.
    """

    # Transpose of the intrinsic matrix K
    K_T = K.T

    # Compute the initial Essential matrix E using E = K^T * F * K
    E = np.dot(K_T, np.dot(F, K))


    # Apply Singular Value Decomposition (SVD) to E to enforce constraints for a valid Essential matrix
    U, S, Vt = np.linalg.svd(E)
    

    
    # Essential matrix constraint: Enforce two singular values to be 1 and the third to be 0
    # This is because an Essential matrix has a rank of 2 and two equal non-zero singular values.

    S_corrected = [1, 1, 0]  # Enforce the two non-zero singular values to be 1 and the third to 0

    
    # Reconstruct the corrected Essential matrix by applying the modified singular values
    new_E = np.dot(U, np.dot(np.diag(S_corrected), Vt))

    
    return new_E  # Return the corrected Essential matrix



# Example Fundamental Matrix and Intrinsic Matrix
# F_opencv = np.array([
#     [-7.56671239e-07, -1.25248835e-05, 2.63295984e-03],
#     [ 1.58888855e-05, -1.14437417e-06, -5.06155563e-03],
#     [-4.35148234e-03,  2.34451568e-03, 1.00000000e+00]
# ])
# K = np.array([[568.99614085, 0., 643.21055941],
#               [0., 568.9883624, 477.98280104],
#               [0., 0., 1.]])

# Compute Essential Matrix
# E = EssentialMatrixFromFundamentalMatrix(F_opencv, K)
# print("Essential Matrix (E):\n", E)

# # Validate Singular Values
# U, S, Vt = np.linalg.svd(E)
# print("Singular values of E:", S)

# Expected Output:
# Singular values of E: [1.0, 1.0, 0.0]

