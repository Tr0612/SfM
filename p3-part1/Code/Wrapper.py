"""
This is a startup script to processes a set of images to perform Structure from Motion (SfM) by
extracting feature correspondences, estimating camera poses, and triangulating 3D points, 
performing PnP and Bundle Adjustment.
"""

import numpy as np
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt

# Import required functions for various steps in the SfM pipeline.

from utils import *
from scipy.spatial.transform import Rotation
# from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from GetInliersRANSAC import GetInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from PnPRANSAC import *
from NonlinearTriangulation import NonlinearTriangulation
from NonlinearPnP import NonlinearPnP
from BuildVisibilityMatrix import *
from BundleAdjustment import BundleAdjustment

################################################################################
# Step 1: Parse all matching files and assign IDs to feature points.
# Each file contains matches between feature points in successive images.
################################################################################
os.chdir(os.path.dirname(__file__))
print("Current Working Directory:", os.getcwd())

file1 = '../Data/Imgs/matching1.txt'
file2 = '../Data/Imgs/matching2.txt'
file3 = '../Data/Imgs/matching3.txt'
file4 = '../Data/Imgs/matching4.txt'
file5 = '../Data/Imgs/matching5.txt'
"""
Assign Unique IDs to feature points across datasets.
"""

"""
The IndexAllFeaturePoints function takes five text files (file1 through file5), each representing feature point matches from different images. It processes each file to:
1. Extract and clean feature point data.
2. Assign unique identifiers (IDs) to each feature point.
3. Ensure that feature points shared across different files are assigned the same ID.
"""

# Check if processed matching files already exist to avoid redundant processing.
if not os.path.exists('../Data/new_matching1.txt'):
    print("\nProcessing Feature Correspondences from matching files...")
    # Process the matching files to assign a unique ID to each feature point.
    # This enables tracking of each point across multiple images in the dataset.
    # Each feature point across the entire dataset will have a unique index.
    match1DF, match2DF, match3DF, match4DF, match5DF = IndexAllFeaturePoints(file1, file2, file3, file4, file5)

else:
    print('Refined Features Indexes Already Exists')
    # Refer utils.py for color definiton
    print(bcolors.WARNING + "Warning: Continuing with the existing Feature Indexes..." + bcolors.ENDC)


################################################################################
# Step 2: Parse matching file for the first pair of cameras (cameras 1 and 2).
# Each file contains matches between feature points in successive images.
################################################################################

# Define the file path and camera indices for parsing keypoints.
file_path = '../Data/new_matching1.txt'
source_camera_index = 1
target_camera_index = 2

# Execute the keypoint parsing for the specified camera pair.
# The output DataFrame provides a structured set of matched keypoints between two images.
ParseKeypoints_DF = ParseKeypoints(file_path, source_camera_index, target_camera_index)

# Extract coordinates for source and target keypoints
# Select specific columns to represent keypoint coordinates in each image
# - 0: Keypoint ID, which uniquely identifies each match
# - 2, 3: X and Y coordinates of the keypoint in the source image
# - 5, 6: X and Y coordinates of the corresponding keypoint in the target image
source_keypoints = ParseKeypoints_DF[[0, 2, 3]]
target_keypoints = ParseKeypoints_DF[[0, 5, 6]]

################################################################################
# Step 3: RANSAC-based outlier rejection.
# Remove outlier based on fundamental matrix. For every eight points, you can
# get one fundamental matrix. But only one fundamental matrix is correct between
# two images. Optimize and find the best fundamental matrix and remove outliers
# based on the features that corresponds to that fundamental matrix.
# This step refines the matches by removing outliers and retaining inliers.
################################################################################

# Use RANSAC to estimate a robust Fundamental matrix (F) that minimizes the impact of outliers.
# Write a function GetInliersRANSAC that removes outliers and compute Fundamental Matrix
# using initial feature correspondences

source_inliers, target_inliers, fundamental_matrix = GetInliersRANSAC(source_keypoints, target_keypoints)
print(bcolors.OKCYAN + "\nFundamental Matrix" + bcolors.OKCYAN)
print(fundamental_matrix, '\n')


F_cv, mask = cv2.findFundamentalMat(source_keypoints.iloc[:, 1:3].to_numpy(),
                                     target_keypoints.iloc[:, 1:3].to_numpy(),
                                     cv2.FM_RANSAC,
                                     ransacReprojThreshold=0.5)
print("OpenCV Fundamental Matrix:\n", F_cv)

source_inliers_np = source_inliers.iloc[:, 1:3].to_numpy()  # Convert to NumPy array (X, Y coordinates)
target_inliers_np = target_inliers.iloc[:, 1:3].to_numpy()  # Convert to NumPy array (X, Y coordinates)

# Scatter plot for visualization
plt.figure(figsize=(8, 6))
plt.scatter(source_inliers_np[:, 0], source_inliers_np[:, 1], label="Source Inliers", c="green")
plt.scatter(target_inliers_np[:, 0], target_inliers_np[:, 1], label="Target Inliers", c="blue")
plt.legend()
plt.title("Inlier Points")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.grid(True)
plt.show()

#################################################################################
# You will write another function 'EstimateFundamentalMatrix' that computes F matrix
# This function is being called by the 'GetInliersRANSAC' function

#################################################################################
# Visualize the final feature correspondences after computing the correct Fundamental Matrix.
# Write a code to print the final feature matches and compare them with the original ones.


################################################################################
# Step 4: Load intrinsic camera matrix K, which contains focal lengths and 
# principal point.
# The calibration file provides intrinsic parameters which are used to calculate
# the Essential matrix.
################################################################################

calib_file = '../Data/Imgs/calibration.txt'
K = process_calib_matrix(calib_file)
print(bcolors.OKCYAN + "\nIntrinsic camera matrix K:" + bcolors.OKCYAN)
print(K, '\n')


################################################################################
# Step 5: Compute Essential Matrix from Fundamental Matrix
################################################################################
E = EssentialMatrixFromFundamentalMatrix(F_cv, K)
print(bcolors.OKCYAN + "\nEssential Matrix E:" + bcolors.OKCYAN)
print(E, '\n')

################################################################################
# Step 6: Extract Camera Poses from Essential Matrix
# Note: You will obtain a set of 4 translation and rotation from this function
################################################################################
Cset, Rset = ExtractCameraPose(E)
print(bcolors.OKCYAN + "\nCset and Rset" + bcolors.OKCYAN)
print(Cset, '\n')
print(Rset, '\n')

################################################################################
# Step 6: Linear Triangulation
################################################################################
# Initialize an empty list to store the 3D points calculated for each camera pose
print(bcolors.OKCYAN + "\nSource Inliers" + bcolors.OKCYAN)
print(source_inliers.shape)
C0 = np.zeros((3,1))    # Camera center at origin
R0 = np.eye(3)

Xset = []

# print(bcolors.OKCYAN + "\nX1 set and X2 set" + bcolors.OKCYAN)
# print(x1Inlier.shape, '\n')
# print(x2Inlier.shape, '\n')
# Iterate over each camera pose in Cset and Rset
for i in range(4):
    # Perform linear triangulation to estimate the 3D points given:
    # - K: Intrinsic camera matrix
    # - np.zeros((3,1)): The initial camera center (assumes origin for the first camera)
    # - np.eye(3): The initial camera rotation (identity matrix, assuming no rotation for the first camera)
    # - Cset[i]: Camera center for the i-th pose
    # - Rset[i]: Rotation matrix for the i-th pose
    # - x1Inlier: Inlier points in the source image
    # - x2Inlier: Corresponding inlier points in the target image
    Xset_i = LinearTriangulation(K, C0,R0, Cset[i], Rset[i], source_inliers, target_inliers)
    Xset.append(Xset_i)

print(bcolors.OKCYAN + "\nXset after Linear Triangulation" + bcolors.OKCYAN)
print(np.array(Xset).shape, '\n')

def validate_epipolar_constraint(E, source_points, target_points):
    """
    Validates the epipolar constraint for corresponding points using the essential matrix.

    Args:
    - E (ndarray): The 3x3 essential matrix.
    - source_points (ndarray): Source points in homogeneous coordinates (Nx2 or Nx3).
    - target_points (ndarray): Target points in homogeneous coordinates (Nx2 or Nx3).

    Returns:
    - None
    """
    # Ensure points are in Nx3 homogeneous form
    if source_points.shape[1] == 2:
        source_hom = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
    else:
        source_hom = source_points

    if target_points.shape[1] == 2:
        target_hom = np.hstack((target_points, np.ones((target_points.shape[0], 1))))
    else:
        target_hom = target_points

    # Compute the epipolar constraint for all points
    errors = np.abs(np.sum(target_hom @ E * source_hom, axis=1))

    # Print average and maximum error
    print("Average Epipolar Error:", np.mean(errors))
    print("Maximum Epipolar Error:", np.max(errors))


# Call the validation function
validate_epipolar_constraint(E, source_inliers, target_inliers)


def verify_reprojection(X3D, x2D, C, R, K, image_path, output_path=None):
    """
    Verifies the reprojection of 3D points onto the image plane and compares
    with the observed 2D points.

    Args:
        X3D (ndarray): 3D points in world coordinates (Nx4, including IDs).
        x2D (ndarray): Observed 2D points in the image plane (Nx3, including IDs).
        C (ndarray): Camera center (3x1).
        R (ndarray): Camera rotation matrix (3x3).
        K (ndarray): Intrinsic camera matrix (3x3).
        image_path (str): Path to the image file.
        output_path (str, optional): Path to save the output visualization. Defaults to None.

    Returns:
        None
    """
    # Load the image
    image = plt.imread(image_path)
    
    # Compute the projection matrix P = K [R | -RC]
    t = -np.dot(R, C)
    P = np.dot(K, np.hstack((R, t.reshape(-1, 1))))
    
    # Prepare 3D points in homogeneous coordinates
    X_homogeneous = X3D[:, 1:]  # Exclude the ID column
    X_homogeneous = np.hstack((X_homogeneous, np.ones((X_homogeneous.shape[0], 1))))  # Add 1 for homogeneous
    
    # Project 3D points onto the image plane
    projected_points = np.dot(P, X_homogeneous.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2:]  # Normalize by depth
    
    # Extract observed 2D points (u, v)
    observed_points = x2D[:, 1:3]  # Exclude the ID column
    
    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.scatter(observed_points[:, 0], observed_points[:, 1], c='green', label='Observed 2D Points', s=10)
    plt.scatter(projected_points[:, 0], projected_points[:, 1], c='red', label='Reprojected 3D Points', s=10)
    plt.legend()
    plt.title("Reprojection Verification")
    plt.xlabel("Image X")
    plt.ylabel("Image Y")

    # Save the plot if an output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Reprojection visualization saved at: {output_path}")

    # Show the plot
    plt.show()

################################################################################
# Call verify_reprojection with appropriate parameters
################################################################################

# Use the triangulated 3D points from the selected camera pose
selected_index = 1  # Change based on your selected camera pose index
X3D = np.array(Xset[selected_index])  # 3D points corresponding to the selected pose
x2D = target_inliers.to_numpy()  # Use the inlier 2D points corresponding to the target camera
C = Cset[selected_index]  # Camera center for the selected pose
R = Rset[selected_index]  # Rotation matrix for the selected pose

# Path to the target image
image_path = "../Data/Imgs/2.jpg"  # Update with your actual image path
output_path = "../Data/Plots/reprojection_verification.png"  # Optional save path

verify_reprojection(X3D, x2D, C, R, K, image_path, output_path)

################################################################################
## Step 7: Plot all points and camera poses
# Write a function: PlotPtsCams that visualizes the 3D points and the estimated camera poses.
################################################################################

# Arguments:
# - Cset: List of camera centers for each pose.
# - Rset: List of rotation matrices for each pose.
# - Xset: List of triangulated 3D points corresponding to each camera pose.
# - SAVE_DIR: Output directory to save the plot.
# - FourCameraPose.png: Filename for the output plot showing 3D points and camera poses.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np

def PlotCameraPts(Cset, Rset, Xset, SAVE_DIR, filename, mode="3D"):
    """
    Visualizes the 3D points and the estimated camera poses.

    Args:
    - Cset (list): List of camera centers (Nx3 numpy arrays).
    - Rset (list): List of rotation matrices (Nx3x3 numpy arrays).
    - Xset (list): List of triangulated 3D points (Nx4 numpy arrays with [ID, X, Y, Z]).
    - SAVE_DIR (str): Output directory to save the plot.
    - filename (str): Filename for the saved plot.
    - mode (str): Visualization mode ("3D" or "2D"). Defaults to "3D".

    Returns:
    - None
    """
    # Ensure SAVE_DIR exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    points_plotted = False
    for X in Xset:
        points = np.array(X)[:, 1:]  # Exclude the ID column
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o', s=1, label="3D Points" if not points_plotted else "")
        points_plotted = True

    # Plot camera centers and orientations
    for i, (C, R) in enumerate(zip(Cset, Rset)):
        # Plot camera center
        ax.scatter(C[0], C[1], C[2], c='red', marker='^', s=50, label=f"Camera {i + 1}")
        
        # Plot camera orientation vectors
        for axis, color, label in zip(range(3), ['r', 'g', 'b'], ['X', 'Y', 'Z']):
            direction = R[:, axis] * 500  # Scale for visualization
            ax.quiver(C[0], C[1], C[2], direction[0], direction[1], direction[2], color=color, label=f"{label}-axis" if i == 0 else "")

    # Set labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Points and Camera Poses")

    # Add legend and save the plot
    ax.legend(loc="upper left", fontsize="small")
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    print(f"Plot saved at: {save_path}")

    # Show the plot
    if mode == "3D":
        plt.show()
    elif mode == "2D":
        plt.close(fig)

# # Example usage
SAVE_DIR = "../Data/Plots/"
# PlotCameraPts(Cset, Rset, Xset, SAVE_DIR, "FourCameraPose.png")

def remove_outliers(Xset, threshold=1000):
    """
    Removes outliers from the 3D points based on a threshold.

    Args:
        Xset (list): List of triangulated 3D points (Nx4 numpy arrays with [ID, X, Y, Z]).
        threshold (float): Threshold for filtering outliers. Points with |X|, |Y|, or |Z| > threshold are removed.

    Returns:
        filtered_Xset (list): List of filtered 3D points.
    """
    filtered_Xset = []
    for X in Xset:
        points = np.array(X)  # Convert to numpy array for filtering
        valid_points = points[
            (np.abs(points[:, 1]) < threshold) &  # Filter X coordinate
            (np.abs(points[:, 2]) < threshold) &  # Filter Y coordinate
            (np.abs(points[:, 3]) < threshold)    # Filter Z coordinate
        ]
        filtered_Xset.append(valid_points)
    return filtered_Xset

# Remove outliers from Xset
threshold = 1000  # Set a threshold for outlier removal
filtered_Xset = remove_outliers(Xset, threshold=threshold)

# Visualize the filtered points
PlotCameraPts(Cset, Rset, filtered_Xset, SAVE_DIR, "FilteredCameraPose.png")



################################################################################
## Step 8: Disambiguate Camera Pose
# Write a function: DisambiguateCameraPose
# DisambiguateCameraPose is called to identify the correct camera pose from multiple
# hypothesized poses. It selects the pose with the most inliers in front of both 
# cameras (i.e., the pose with the most consistent triangulated 3D points).
################################################################################

## Disambiguate camera poses
# Arguments:
# - Cset: List of candidate camera centers for each pose.
# - Rset: List of candidate rotation matrices for each pose.
# - Xset: List of sets of triangulated 3D points for each camera pose.
# Returns:
# - C: The selected camera center after disambiguation.
# - R: The selected rotation matrix after disambiguation.
# - X: The triangulated 3D points for the selected camera pose.
# - selectedIdx: The index of the selected camera pose within the original candidate sets.
C, R, X, selectedIdx = DisambiguateCameraPose(Cset, Rset, filtered_Xset)
print(bcolors.OKCYAN + "\nC" + bcolors.OKCYAN)
print(C.shape, '\n')
print(bcolors.OKCYAN + "\nR" + bcolors.OKCYAN)
print(R.shape, '\n')
print(bcolors.OKCYAN + "\nSelected Index" + bcolors.OKCYAN)
print(selectedIdx, '\n')
# Plot the selected camera pose with its 3D points
# This plot shows the selected camera center, orientation, and the corresponding 3D points.
# Arguments:
# - [C]: List containing the selected camera center (wrapping it in a list for compatibility with PlotPtsCams).
# - [R]: List containing the selected rotation matrix.
# - [X]: List containing the 3D points for the selected camera pose.
# - SAVE_DIR: Output directory to save the plot.
# - OneCameraPoseWithPoints.png: Filename for the output plot showing both the camera pose and 3D points.
# - show_pos=True: Enables the display of the camera pose.

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def PlotPtsCams(Cset, Rset, Xset, SAVE_DIR, filename, show_pos=True):
    """
    Visualizes the 3D points and the selected camera pose.

    Args:
    - Cset (list): List of camera centers (each 3x1 numpy array).
    - Rset (list): List of rotation matrices (each 3x3 numpy array).
    - Xset (list): List of triangulated 3D points (each Nx4 numpy array with [ID, X, Y, Z]).
    - SAVE_DIR (str): Output directory to save the plot.
    - filename (str): Filename for the saved plot.
    - show_pos (bool): Flag to enable the display of camera pose. Defaults to True.

    Returns:
    - None
    """
    # Ensure SAVE_DIR exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    for X in Xset:
        points = np.array(X)[:, 1:]  # Exclude the ID column
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o', s=1, label="3D Points")

    # Plot camera center
    for i, C in enumerate(Cset):
        ax.scatter(C[0], C[1], C[2], c='red', marker='^', s=50, label=f"Camera Center {i + 1}")

        # Plot camera orientation vectors if enabled
        if show_pos and i < len(Rset):
            R = Rset[i]
            for axis, color, label in zip(range(3), ['r', 'g', 'b'], ['X', 'Y', 'Z']):
                direction = R[:, axis] * 0.5  # Scale for visualization
                ax.quiver(C[0], C[1], C[2], direction[0], direction[1], direction[2], color=color, label=f"{label}-axis")

    # Set labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("Selected Camera Pose with 3D Points")

    # Save the plot
    save_path = os.path.join(SAVE_DIR, filename)
    plt.legend(loc="best")
    plt.savefig(save_path)
    print(f"Plot saved at: {save_path}")

    # Show the plot
    plt.show()

# Call the function with the selected camera pose and its 3D points
SAVE_DIR = "../Data/Plots/"
PlotPtsCams([C], [R], [X], SAVE_DIR, "OneCameraPoseWithPoints.png")


################################################################################
## Step 9: Non-Linear Triangulation
# Write a function: NonLinearTriangulation
# Inputs:
# - K: Intrinsic camera matrix of the first camera (3x3).
# - C0, R0: Translation (3x1) and rotation (3x3) of the first camera.
# - Cseti, Rseti: Translations and rotations of other cameras in a list.
# - x1set, x2set: Sets oC0f 2D points in each image for triangulation.
# - X0: Initial 3D points for optimization.
# Output:
# - Returns optimized 3D points after minimizing reprojection error.
# NonlinearTriangulation(K, C0, R0, Cseti, Rseti, x1set, x2set, X0):
################################################################################
x1set = source_inliers.iloc[:, 1:3].to_numpy()  # Extracting x, y coordinates
X0 = np.array(Xset[selectedIdx])[:, 1:4]
# 2D points in other camera images
x2set = [target_inliers.iloc[:, 1:3].to_numpy()]
print("K shape:", K.shape)
print("C0 shape:", Cset[0].shape)
print("R0 shape:", Rset[0].shape)
print("Number of cameras:", len(Cset), len(Rset))
print("x1Inlier shape:", x1set.shape)
# print("x2Inlier shape:", x2set.shape)

# X_nl = NonlinearTriangulation(
#     K,      # Intrinsic matrix
#     Cset[0],     # Camera center for the first camera
#     Rset[0],     # Rotation for the first camera
#     Cset[1:],   # List of other camera centers
#     Rset[1:],   # List of other camera rotations
#     x1set,  # Inlier points in the first camera
#     x2set,  # Inlier points in other cameras
#     X0      # Initial 3D points
# )

# Output the optimized 3D points
print("\nOptimized 3D Points after Nonlinear Triangulation:")
# np.save("X_nl.npy",X_nl)
X_nl = np.load("X_nl.npy")
print(X_nl.shape)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
def remove_outliers(X_3D, x1_2D, x2_2D_list, K, C0, R0, Cset, Rset, threshold_reproj=100, spatial_limit=3000):
    """
    Removes outliers from 3D points based on reprojection errors and spatial constraints.
    
    Args:
        X_3D (ndarray): 3D points (Nx3).
        x1_2D (ndarray): Observed 2D points in the source image (Nx2).
        x2_2D_list (list): List of observed 2D points in the target images (each Nx2).
        K (ndarray): Intrinsic camera matrix (3x3).
        C0 (ndarray): Translation vector for the first camera (3x1).
        R0 (ndarray): Rotation matrix for the first camera (3x3).
        Cset (list): List of translation vectors for other cameras (each 3x1).
        Rset (list): List of rotation matrices for other cameras (each 3x3).
        threshold_reproj (float): Maximum allowable reprojection error (default: 100 pixels).
        spatial_limit (float): Maximum allowable spatial distance from the origin (default: 3000 units).

    Returns:
        X_3D_filtered, x1_2D_filtered, x2_2D_filtered: Filtered points.
    """
    reprojection_errors = []
    valid_indices = []

    for i, X_world in enumerate(X_3D):
        X_hom = np.append(X_world, 1)  # Homogeneous 3D point

        # Compute reprojection for the first camera
        P0 = K @ R0 @ np.hstack((np.eye(3), -C0))
        proj1 = P0 @ X_hom
        proj1 /= proj1[2]  # Normalize to pixel coordinates
        error1 = np.linalg.norm(x1_2D[i] - proj1[:2])

        # Compute reprojection for all other cameras
        total_error = error1
        for j in range(len(Cset)):  # Ensure we loop through valid camera indices
            if j >= len(x2_2D_list):  # Check if target 2D points exist for this camera
                break

            x2 = x2_2D_list[j]  # Get corresponding target 2D points for camera j
            Cj = Cset[j]
            Rj = Rset[j]
            Pj = K @ Rj @ np.hstack((np.eye(3), -Cj))

            projj = Pj @ X_hom
            projj /= projj[2]  # Normalize to pixel coordinates
            error2 = np.linalg.norm(x2[i] - projj[:2])
            total_error += error2

        reprojection_errors.append(total_error)

        # Apply reprojection and spatial constraints
        if total_error < threshold_reproj and np.linalg.norm(X_world) < spatial_limit:
            valid_indices.append(i)

    # Filter points
    X_3D_filtered = X_3D[valid_indices]
    x1_2D_filtered = x1_2D[valid_indices]
    x2_2D_filtered = [x2[valid_indices] for x2 in x2_2D_list[:len(Cset)]]

    return X_3D_filtered, x1_2D_filtered, x2_2D_filtered

# Example usage:
threshold_reproj = 100  # Set your desired reprojection error threshold
spatial_limit = 3000    # Set your desired spatial distance limit

# X_3D_filtered, x1_2D_filtered, x2_2D_filtered = remove_outliers(
#     X_nl,               # Optimized 3D points
#     x1set,              # 2D inliers from the source image
#     [x2set],            # List of 2D inliers from target images
#     K, Cset[0], Rset[0], Cset[1:], Rset[1:], threshold_reproj, spatial_limit
# )

# print(f"Original 3D Points: {len(X_nl)}, Filtered 3D Points: {len(X_3D_filtered)}")



# Load the optimized 3D points

# Function to visualize 3D points
def visualize_3d_points(points, title="Optimized 3D Points"):
    """
    Visualizes the 3D points in a 3D scatter plot.
    
    Args:
    - points: Numpy array of shape (N, 3), representing 3D points.
    - title: Title for the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1, label="3D Points")
    
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()
    plt.show()

# Function to calculate and visualize reprojection error
def calculate_reprojection_error(X3D, x2D, K, C, R, image_path=None):
    """
    Projects 3D points back to 2D image plane and calculates reprojection error.
    Visualizes the comparison between reprojected points and observed points.
    
    Args:
    - X3D: Optimized 3D points, shape (N, 3).
    - x2D: Original 2D points, shape (N, 2).
    - K: Intrinsic camera matrix, shape (3, 3).
    - C: Camera center, shape (3, 1).
    - R: Rotation matrix, shape (3, 3).
    - image_path: Path to the image for visualization, if available.
    """
    # Projection matrix
    t = -np.dot(R, C)
    P = np.dot(K, np.hstack((R, t)))
    
    # Homogeneous 3D points
    X_hom = np.hstack((X3D, np.ones((X3D.shape[0], 1))))
    
    # Project points
    x_proj_hom = np.dot(P, X_hom.T).T
    x_proj = x_proj_hom[:, :2] / x_proj_hom[:, 2:3]
    
    # Reprojection error
    reproj_error = np.linalg.norm(x_proj - x2D, axis=1)
    avg_error = np.mean(reproj_error)
    max_error = np.max(reproj_error)
    
    # Visualization
    if image_path:
        image = plt.imread(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.scatter(x2D[:, 0], x2D[:, 1], c='green', s=10, label="Observed 2D Points")
        plt.scatter(x_proj[:, 0], x_proj[:, 1], c='red', s=10, label="Reprojected 3D Points")
        plt.title(f"Reprojection Visualization\nAvg Error: {avg_error:.2f}, Max Error: {max_error:.2f}")
        plt.legend()
        plt.show()
    
    return avg_error, max_error

# Visualize the optimized 3D points
visualize_3d_points(X_nl[:, :3], title="Optimized 3D Points After Nonlinear Triangulation")

# Assume C, R, K, and x2D are available
C = np.array([[0], [0], [0]])  # Example camera center
R = np.eye(3)  # Example rotation matrix
K = np.array([[568.99614085, 0, 643.21055941],
              [0, 568.9883624, 477.98280104],
              [0, 0, 1]])  # Example intrinsic matrix
x2D = np.random.rand(X_nl.shape[0], 2) * [1280, 720]  # Replace with actual 2D points

# Calculate and visualize reprojection error
avg_error, max_error = calculate_reprojection_error(X_nl[:, :3], x2D, K, C, R, image_path="../Data/Imgs/2.jpg")

# Print reprojection errors
print(f"Average Reprojection Error: {avg_error}")
print(f"Maximum Reprojection Error: {max_error}")


################################################################################
# Step 10: PnPRANSAC
# PnPRANSAC: Function to perform PnP using RANSAC to find the best camera pose 
# with inliers
################################################################################
# Prepare 3D points (Xset)
# Prepare 3D points (Xset)
# Add IDs as the first column to X_nl
Xset = np.hstack((np.arange(X_nl.shape[0]).reshape(-1, 1), X_nl))  # Add IDs as the first column

# Prepare 2D points (xset)
xset = np.hstack((np.arange(source_inliers.shape[0]).reshape(-1, 1), source_inliers))  # Add IDs as the first column

# Validate shapes
print("Xset shape:", Xset.shape)  # Expected (451, 4)
print("xset shape:", xset.shape)  # Expected (451, 3)

# Call PnPRANSAC
Cnew, Rnew, Inliers = PnPRANSAC(Xset, xset, K)

# Output results
print("Estimated Camera Center (Cnew):", Cnew)
print("Estimated Rotation Matrix (Rnew):\n", Rnew)
print("Number of Inliers:", len(Inliers))


# Visualize Reprojection
def visualize_reprojection(X3D, x2D, C, R, K, image_path, output_path=None):
    """
    Visualizes the reprojection of 3D points onto the image plane and compares
    with the original 2D points.

    Args:
        X3D (ndarray): 3D points in the world coordinate system (Nx4).
        x2D (ndarray): Original 2D points in the image coordinate system (Nx3).
        C (ndarray): Camera center (3x1).
        R (ndarray): Rotation matrix (3x3).
        K (ndarray): Intrinsic camera matrix (3x3).
        image_path (str): Path to the image file.
        output_path (str, optional): Path to save the output visualization. Defaults to None.

    Returns:
        None
    """
    # Load the image
    image = plt.imread(image_path)
    
    # Compute projection matrix P = K [R | -RC]
    t = -np.dot(R, C)
    P = np.dot(K, np.hstack((R, t.reshape(-1, 1))))
    
    # Convert 3D points to homogeneous coordinates (Nx4)
    X_homogeneous = np.hstack((X3D[:, :3], np.ones((X3D.shape[0], 1))))
    
    # Project the 3D points into the 2D image plane
    projected_points = np.dot(P, X_homogeneous.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]  # Normalize by depth
    
    # Original 2D points
    original_points = x2D[:, 1:3]  # Extract x and y coordinates
    
    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.scatter(original_points[:, 0], original_points[:, 1], c='green', label='Original 2D Points', s=10)
    plt.scatter(projected_points[:, 0], projected_points[:, 1], c='red', label='Reprojected 3D Points', s=10)
    plt.legend()
    plt.title("Reprojection Debug Visualization")
    plt.xlabel("Image X")
    plt.ylabel("Image Y")

    # Save the visualization if output_path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved at: {output_path}")

    # Show the plot
    plt.show()

# Example Usage
image_path = "../Data/Imgs/2.jpg"  # Replace with your image path
output_path = "../Data/Plots/reprojection_debug.png"  # Optional save path

# Ensure `Xset` and `xset` have compatible dimensions for reprojection
visualize_reprojection(Xset, xset, Cnew, Rnew, K, image_path, output_path)
# print("Inliers:", Inliers)
# # Call PnPRANSAC to estimate camera pose
# Cnew, Rnew, Inliers = PnPRANSAC(Xset_df, xset_df, K)
################################################################################
# Step 11: NonLinearPnP
# NonLinearPnP: Refines the camera pose (position and orientation) using non-linear 
# optimization to minimize the reprojection error between observed 2D points and 
# projected 3D points.
################################################################################
# Similarly x_inliers should be (M,3)

# Prepare inputs for NonlinearPnP
Xs = np.hstack((np.arange(Xset.shape[0]).reshape(-1, 1), Xset))  # Add IDs as the first column
xs = np.hstack((np.arange(source_inliers.shape[0]).reshape(-1, 1), source_inliers))  # Add IDs as the first column

# Initial estimates for camera position and rotation
Cnew = np.array(Cnew).reshape(3, 1)  # Ensure it's a column vector
Rnew = np.array(Rnew)  # Ensure it's a proper 3x3 matrix

# Validate the inputs
print("Xs shape:", Xs.shape)  # Expected (N, 4) with IDs
print("xs shape:", xs.shape)  # Expected (N, 3) with IDs
print("K shape:", K.shape)  # Expected (3, 3)
print("Initial Camera Center (Cnew):", Cnew)
print("Initial Rotation Matrix (Rnew):\n", Rnew)

# Call NonlinearPnP
Copt, Ropt = NonlinearPnP(Xs, xs, K, Cnew, Rnew)

# Validate outputs
print("\nOptimized Camera Center (Copt):", Copt)
print("Optimized Rotation Matrix (Ropt):\n", Ropt)

# Check if the rotation matrix is valid
print("Orthogonality Check (Ropt x Ropt^T):\n", np.dot(Ropt, Ropt.T))
print("Determinant of Ropt:", np.linalg.det(Ropt))


image_path = "../Data/Imgs/2.jpg"  # Path to the image
output_path = "../Data/Plots/reprojection_visualization.png"  # Optional output path

import matplotlib.pyplot as plt
import numpy as np

def visualize_reprojection(X_3D, x_2D_observed, K, C, R, image_path, output_path=None):
    """
    Visualizes the reprojection of 3D points onto the image plane and compares
    with the observed 2D points.
    
    Args:
    - X_3D (ndarray): Optimized 3D points (Nx3).
    - x_2D_observed (ndarray): Observed 2D points in the image plane (Nx2).
    - K (ndarray): Intrinsic camera matrix (3x3).
    - C (ndarray): Optimized camera center (3x1).
    - R (ndarray): Optimized rotation matrix (3x3).
    - image_path (str): Path to the image file.
    - output_path (str, optional): Path to save the visualization. Defaults to None.

    Returns:
    - None
    """
    # Load the image
    image = plt.imread(image_path)
    
    # Prepare the projection matrix: P = K * [R | -R*C]
    t = -np.dot(R, C)
    P = np.dot(K, np.hstack((R, t)))
    
    # Convert 3D points to homogeneous coordinates
    X_homogeneous = np.hstack((X_3D, np.ones((X_3D.shape[0], 1))))
    
    # Project 3D points to the image plane
    x_2D_projected = np.dot(P, X_homogeneous.T).T  # (Nx3)
    x_2D_projected = x_2D_projected[:, :2] / x_2D_projected[:, 2:3]  # Normalize to (u, v)
    
    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.scatter(x_2D_observed[:, 0], x_2D_observed[:, 1], c='green', label='Observed 2D Points', s=15)
    plt.scatter(x_2D_projected[:, 0], x_2D_projected[:, 1], c='red', label='Reprojected 2D Points', s=15)
    plt.legend()
    plt.title("Reprojection Visualization")
    plt.xlabel("Image X")
    plt.ylabel("Image Y")
    
    # Save the visualization if an output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved at: {output_path}")
    
    # Show the plot
    plt.show()

# Example usage:
image_path = "../Data/Imgs/2.jpg"  # Path to the image
output_path = "../Data/Plots/reprojection_visualization.png"  # Optional output path

# Xs: Optimized 3D points (Nx4 with IDs), use only the 3D coordinates
X_3D = Xs[:, 1:4]

# xs: Observed 2D points (Nx3 with IDs), use only the 2D coordinates
x_2D_observed = xs[:, 1:3]

visualize_reprojection(X_3D, x_2D_observed, K, Copt, Ropt, image_path, output_path)


# Xs: Optimized 3D points (Nx4 with IDs), use only the 3D coordinates
X_3D = Xs[:, 1:4]

# xs: Observed 2D points (Nx3 with IDs), use only the 2D coordinates
x_2D_observed = xs[:, 1:3]

visualize_reprojection(X_3D, x_2D_observed, K, Copt, Ropt, image_path, output_path)
################################################################################
# Step 12: BuildVisibilityMatrix
# BuildVisibilityMatrix: BuildVisibilityMatrix: Constructs a sparse visibility 
# matrix for bundle adjustment. This matrix indicates which parameters affect 
# each observation in the optimization process.
################################################################################
# Example setup for visibility matrix
# Number of cameras
n_cameras = 2

# Extract 3D points from X_nl
X_3D = X_nl[:, 1:4]  # Assuming X_nl has [ID, X, Y, Z], extract only [X, Y, Z]
n_points = X_3D.shape[0]

print(f"X_nl:\n{X_nl}")

# Handle camera indices and point indices for Camera 0 (source_inliers)
camera_indices_0 = np.zeros(len(source_inliers), dtype=int)  # All observations belong to Camera 0
point_indices_0 = source_inliers.iloc[:, 0].astype(int).to_numpy()  # First column contains Point IDs

# Handle camera indices and point indices for Camera 1 (target_inliers)
camera_indices_1 = np.ones(len(target_inliers), dtype=int)  # All observations belong to Camera 1
point_indices_1 = target_inliers.iloc[:, 0].astype(int).to_numpy()  # First column contains Point IDs

# Concatenate camera indices and point indices
camera_indices = np.concatenate([camera_indices_0, camera_indices_1])
point_indices = np.concatenate([point_indices_0, point_indices_1])

# Map point indices to 0-based indexing for points in X_3D
unique_ids = X_nl[:, 0].astype(int)  # Extract IDs from X_nl's first column
id_to_index = {id_val: idx for idx, id_val in enumerate(unique_ids)}

print("Unique IDs in X_nl:", unique_ids)
print("Point IDs in source_inliers:", point_indices_0)
print("Point IDs in target_inliers:", point_indices_1)
print("ID 11 in unique_ids:", 11 in unique_ids)

# Filter valid observations based on unique_ids in X_nl
valid_point_indices = []
valid_camera_indices = []
valid_xall = []

# Create xall for both cameras
xall_0 = source_inliers.iloc[:, 1:3].to_numpy()  # Extract [u, v] from source_inliers
xall_1 = target_inliers.iloc[:, 1:3].to_numpy()  # Extract [u, v] from target_inliers
xall_combined = np.vstack([xall_0, xall_1])

for cam_idx, pt_idx, obs in zip(camera_indices, point_indices, xall_combined):
    if pt_idx in id_to_index:
        valid_point_indices.append(id_to_index[pt_idx])  # Map to 0-based index
        valid_camera_indices.append(cam_idx)
        valid_xall.append(obs)

# Update camera_indices, point_indices, and xall to valid observations
camera_indices = np.array(valid_camera_indices)
point_indices = np.array(valid_point_indices)
xall = np.array(valid_xall)

# Validate dimensions
assert len(camera_indices) == len(point_indices), "Mismatch between camera_indices and point_indices."
assert len(camera_indices) == xall.shape[0], "Mismatch between camera_indices and xall."

# Final validation and output
print(f"Camera Indices Shape: {camera_indices.shape}")
print(f"Point Indices Shape: {point_indices.shape}")
print(f"xall Shape: {xall.shape}")

 # (2N, 2)
assert len(camera_indices) == len(point_indices), "Mismatch between camera_indices and point_indices"
assert len(camera_indices) == xall.shape[0], "Mismatch between camera_indices and xall"

# Now camera_indices, point_indices, and xall should match:
# len(camera_indices) == len(point_indices) == xall.shape[0]

# Build visibility matrix
visibility_matrix = BuildVisibilityMatrix(n_cameras, n_points, camera_indices, point_indices)


# Outputs
print("Visibility Matrix Shape:", visibility_matrix.shape)
print("Sample Visibility Matrix:", visibility_matrix.toarray())


################################################################################
# Step 13: BundleAdjustment
# BundleAdjustment: Refines camera poses and 3D point positions to minimize the 
# reprojection error for a set of cameras and 3D points using non-linear 
# optimization.
################################################################################


# Assuming you have R0, C0 for camera 0, and R1, C1 for camera 1 (if needed)
Call = [
    np.zeros((3, 1)),   # First camera (at the origin)
    Copt,               # Second camera (optimized from NonlinearPnP)           # Fourth camera
]

Rall = [
    np.eye(3),          # First camera (identity rotation)
    Ropt,               # Second camera (optimized from NonlinearPnP)             # Fourth camera
]       # Adjust as needed

camera_params = []
for C, R in zip(Call, Rall):
    q = Rotation.from_matrix(R).as_quat()  # Convert rotation matrix to quaternion
    camera_params.append(np.hstack((C.ravel(), q)))  # Combine translation and quaternion
camera_params = np.array(camera_params).ravel()  # Flatten
print("Camera Params Shape:", camera_params.shape)

CoptAll, RoptAll, XoptAll = BundleAdjustment(
    Call=Call,
    Rall=Rall,
    Xall=X_3D,
    K=K,
    sparseVmatrix=visibility_matrix,
    n_cameras=n_cameras,
    n_points=n_points,
    camera_indices=camera_indices,
    point_indices=point_indices,
    xall=xall
)

# Output optimized results
print("Optimized Camera Centers:", CoptAll)
print("Optimized Rotation Matrices:", RoptAll)
print("Optimized 3D Points:\n", XoptAll)
