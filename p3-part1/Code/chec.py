# import numpy as np
# import cv2
# from utils import *
# # from EstimateFundamentalMatrix import EstimateFundamentalMatrix
# from GetInliersRANSAC import GetInliersRANSAC

# def VerifyFundamentalMatrixWithOpenCV(x1All, x2All, T=0.3, debug=False):
#     """
#     Verify the computed Fundamental matrix using OpenCV's findFundamentalMat function.
    
#     Args:
#         x1All (DataFrame): Source image points with IDs and (x, y) coordinates.
#         x2All (DataFrame): Target image points with IDs and (x, y) coordinates.
#         T (float): Threshold for RANSAC-based outlier rejection. Default is 0.3.
#         debug (bool): Flag to enable debugging information.
    
#     Returns:
#         F_opencv (ndarray): Fundamental matrix estimated by OpenCV.
#         inlier_mask (ndarray): Mask indicating inliers based on RANSAC.
#         x1Inlier_opencv (DataFrame): Inlier points in the source image.
#         x2Inlier_opencv (DataFrame): Inlier points in the target image.
#     """
#     if debug:
#         print("\n------- BEGIN VerifyFundamentalMatrixWithOpenCV -------")
#         print("Shape of x1All:", x1All.shape)
#         print("Shape of x2All:", x2All.shape)

#     # Extract coordinates for OpenCV
#     x1_coords = x1All[[2, 3]].to_numpy()  # Columns 2 and 3 correspond to x and y
#     x2_coords = x2All[[5, 6]].to_numpy()  # Columns 5 and 6 correspond to x and y

#     # Use OpenCV to estimate the Fundamental Matrix
#     F_opencv, inlier_mask = cv2.findFundamentalMat(
#         x1_coords, x2_coords, method=cv2.FM_RANSAC, ransacReprojThreshold=T
#     )
    
#     if F_opencv is None or inlier_mask is None:
#         print("OpenCV failed to estimate a Fundamental Matrix.")
#         return None, None, pd.DataFrame(), pd.DataFrame()

#     # Filter inliers using the inlier mask
#     inlier_indices = np.where(inlier_mask.flatten() == 1)[0]
#     x1Inlier_opencv = x1All.iloc[inlier_indices]
#     x2Inlier_opencv = x2All.iloc[inlier_indices]

#     if debug:
#         print("\n=== OpenCV Fundamental Matrix ===")
#         print(F_opencv)
#         print("\nNumber of inliers detected by OpenCV:", len(inlier_indices))
#         print("--- End of Verification ---")

#     return F_opencv, inlier_mask, x1Inlier_opencv, x2Inlier_opencv

# # Example Usage in the Pipeline
# if __name__ == "__main__":
#     # Load your parsed keypoints for the first image pair
#     os.chdir(os.path.dirname(__file__))
#     print("Current Working Directory:", os.getcwd())
#     file_path = '../Data/new_matching1.txt'
#     source_camera_index = 1
#     target_camera_index = 2
#     ParseKeypoints_DF = ParseKeypoints(file_path, source_camera_index, target_camera_index)

#     # Extract source and target points
#     source_keypoints = ParseKeypoints_DF[[0, 2, 3]]
#     target_keypoints = ParseKeypoints_DF[[0, 5, 6]]

#     # Verify the Fundamental Matrix
#     F_opencv, inlier_mask, x1Inlier_opencv, x2Inlier_opencv = VerifyFundamentalMatrixWithOpenCV(
#         source_keypoints, target_keypoints, T=0.3, debug=True
#     )

#     # Compare with your implementation
#     F_custom = EstimateFundamentalMatrix(
#         source_keypoints[[2, 3]].to_numpy(), target_keypoints[[5, 6]].to_numpy()
#     )
#     print("\n=== Custom Implementation Fundamental Matrix ===")
#     print(F_custom)

import numpy as np

def verify_essential_matrix(E, K, points1, points2):
    """
    Verifies the Essential Matrix.

    Args:
        E (ndarray): Essential Matrix (3x3).
        K (ndarray): Intrinsic camera matrix (3x3).
        points1 (ndarray): Points from image 1 (Nx2).
        points2 (ndarray): Points from image 2 (Nx2).

    Returns:
        None
    """
    print("\n=== Verifying Essential Matrix ===")

    # Step 1: Check the SVD of E
    U, S, Vt = np.linalg.svd(E)
    print("Singular Values of E:", S)
    print("Condition: Two singular values should be equal, and one should be zero.")

    # Ensure the rank of E is 2
    S_corrected = np.diag([S[0], S[1], 0])
    E_corrected = U @ S_corrected @ Vt
    print("\nCorrected Essential Matrix (Enforcing Rank-2):\n", E_corrected)

    # Step 2: Epipolar Constraint Validation
    # Convert points to homogeneous coordinates
    points1_hom = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2_hom = np.hstack((points2, np.ones((points2.shape[0], 1))))

    # Compute epipolar constraint for all point pairs
    errors = []
    for x1, x2 in zip(points1_hom, points2_hom):
        error = np.abs(x2 @ E @ x1.T)
        errors.append(error)
    avg_error = np.mean(errors)
    print(f"\nAverage Epipolar Constraint Error: {avg_error:.6f}")
    print("Ideal: Close to zero.")

    # Step 3: Decompose Essential Matrix into Rotation and Translation
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t = U[:, 2]  # Translation is the third column of U
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Ensure proper rotations (determinant of R must be +1)
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    print("\nPossible Rotations (R1, R2):\n", R1, "\n", R2)
    print("Translation (t):\n", t)

    print("\n=== End of Verification ===")


# Example Usage
E = np.array([[-0.02024298, -0.61320295, -0.27045225],
              [ 0.79948729, -0.07664602,  0.57190476],
              [ 0.18240537, -0.72027465, -0.16196006]])

K = np.array([[568.99614085, 0, 643.21055941],
              [0, 568.9883624, 477.98280104],
              [0, 0, 1]])

# Sample corresponding points (replace with actual points)
points1 = np.array([[100, 200], [150, 250], [300, 400]])
points2 = np.array([[102, 202], [152, 252], [302, 402]])

verify_essential_matrix(E, K, points1, points2)