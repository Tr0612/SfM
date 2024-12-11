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
# from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from GetInliersRANSAC import GetInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from PnPRANSAC import *
from NonlinearTriangulation import NonlinearTriangulation
from NonlinearPnP import NonlinearPnP
# from BuildVisibilityMatrix import *
# from BundleAdjustment import BundleAdjustment

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
#################################################################################
# You will write another function 'EstimateFundamentalMatrix' that computes F matrix
# This function is being called by the 'GetInliersRANSAC' function
#################################################################################

# Visualize the final feature correspondences after computing the correct Fundamental Matrix.
# Write a code to print the final feature matches and compare them with the original ones.
import matplotlib.pyplot as plt
import numpy as np

def draw_matches(image1_path, image2_path, keypoints1, keypoints2, inliers1=None, inliers2=None, output_path=None):
    """
    Draws feature matches (lines between matched points) and compares original matches with inliers.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        keypoints1 (ndarray): Original keypoints in the first image (Nx2).
        keypoints2 (ndarray): Original keypoints in the second image (Nx2).
        inliers1 (ndarray): Inlier keypoints in the first image (Mx2, optional).
        inliers2 (ndarray): Inlier keypoints in the second image (Mx2, optional).
        output_path (str): Path to save the output plot (optional).
    """
    # Load images
    image1 = plt.imread(image1_path)
    image2 = plt.imread(image2_path)

    # Combine the two images side-by-side for visualization
    combined_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)
    combined_image[:image1.shape[0], :image1.shape[1]] = image1
    combined_image[:image2.shape[0], image1.shape[1]:] = image2

    # Adjust keypoints for the combined image
    keypoints2_adjusted = keypoints2.copy()
    keypoints2_adjusted[:, 0] += image1.shape[1]

    # Plot original matches
    plt.figure(figsize=(15, 8))
    plt.imshow(combined_image)
    for kp1, kp2 in zip(keypoints1, keypoints2_adjusted):
        plt.plot([kp1[0], kp2[0]], [kp1[1], kp2[1]], color="red", linestyle="--", linewidth=0.5)
    plt.scatter(keypoints1[:, 0], keypoints1[:, 1], c="yellow", label="Original Matches")
    plt.scatter(keypoints2_adjusted[:, 0], keypoints2_adjusted[:, 1], c="yellow")
    plt.title("Original Feature Matches")
    plt.legend()

    if output_path:
        plt.savefig(f"{output_path}_original_matches.png")
    plt.show()

    # Plot inlier matches (if available)
    if inliers1 is not None and inliers2 is not None:
        # Adjust inliers for the combined image
        inliers2_adjusted = inliers2.copy()
        inliers2_adjusted[:, 0] += image1.shape[1]

        plt.figure(figsize=(15, 8))
        plt.imshow(combined_image)
        for kp1, kp2 in zip(inliers1, inliers2_adjusted):
            plt.plot([kp1[0], kp2[0]], [kp1[1], kp2[1]], color="blue", linewidth=1)
        plt.scatter(inliers1[:, 0], inliers1[:, 1], c="green", label="Inlier Matches")
        plt.scatter(inliers2_adjusted[:, 0], inliers2_adjusted[:, 1], c="green")
        plt.title("Inlier Feature Matches (After RANSAC)")
        plt.legend()

        if output_path:
            plt.savefig(f"{output_path}_inlier_matches.png")
        plt.show()

    # Print comparison of original matches and inliers
    print("\nComparison of Matches:")
    print(f"Total Original Matches: {len(keypoints1)}")
    print(f"Total Inlier Matches: {len(inliers1) if inliers1 is not None else 0}")


# Convert keypoints from DataFrame to numpy arrays for visualization
keypoints1 = source_keypoints.iloc[:, 1:].to_numpy()  # Original matches (source image)
keypoints2 = target_keypoints.iloc[:, 1:].to_numpy()  # Original matches (target image)

# Convert inliers from DataFrame to numpy arrays for visualization
inliers1 = source_inliers.iloc[:, 1:].to_numpy()  # Inlier matches (source image)
inliers2 = target_inliers.iloc[:, 1:].to_numpy()  # Inlier matches (target image)

# Path to images (replace with actual paths)
image1_path = "../Data/Imgs/1.jpg"
image2_path = "../Data/Imgs/2.jpg"
image3_path = "../Data/Imgs/3.jpg"

# Visualize and compare matches
draw_matches(
    image1_path,
    image2_path,
    keypoints1,
    keypoints2,
    inliers1,
    inliers2,
    output_path="../Data/Plots/match_comparison"
)
draw_matches(
    image1_path,
    image3_path,
    keypoints1,
    keypoints2,
    inliers1,
    inliers2,
    output_path="../Data/Plots/match_comparison"
)

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
E = EssentialMatrixFromFundamentalMatrix(fundamental_matrix, K)
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
Xset = []
x1Inlier = inliers1    # Inlier matches (source image)x
x2Inlier = inliers2
print(bcolors.OKCYAN + "\nX1 set and X2 set" + bcolors.OKCYAN)
print(x1Inlier.shape, '\n')
print(x2Inlier.shape, '\n')
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
    Xset_i = LinearTriangulation(K, np.zeros((3,1)), np.eye(3), Cset[i], Rset[i], source_inliers, target_inliers)
    Xset.append(Xset_i)

print(bcolors.OKCYAN + "\nXset after Linear Triangulation" + bcolors.OKCYAN)
print(np.array(Xset).shape, '\n')

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

def PlotCameraPts(Cset, Rset, Xset, SAVE_DIR, filename, mode="2D"):
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    for X in Xset:
        points = np.array(X)[:, 1:]  # Exclude the ID column
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o', s=1, label="3D Points")

    # Plot camera centers
    for i, C in enumerate(Cset):
        ax.scatter(C[0], C[1], C[2], c='red', marker='^', s=50, label=f"Camera {i + 1}")

    # Plot camera orientation vectors
    for i, (C, R) in enumerate(zip(Cset, Rset)):
        for axis, color, label in zip(range(3), ['r', 'g', 'b'], ['X', 'Y', 'Z']):
            direction = R[:, axis] * 0.5  # Scale for visualization
            ax.quiver(C[0], C[1], C[2], direction[0], direction[1], direction[2], color=color, label=f"{label}-axis (Cam {i+1})")

    # Set labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Points and Camera Poses")

    # Save the plot
    save_path = os.path.join(SAVE_DIR, filename)
    plt.legend(loc="best")
    plt.savefig(save_path)
    print(f"Plot saved at: {save_path}")

    # Show the plot
    if mode == "3D":
        plt.show()
    elif mode == "2D":
        plt.close(fig)

SAVE_DIR = "../Data/Plots/"

PlotCameraPts(Cset, Rset, Xset, SAVE_DIR, "FourCameraPose.png")


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
C, R, X, selectedIdx = DisambiguateCameraPose(Cset, Rset, Xset)
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
print("K shape:", K.shape)
print("C0 shape:", Cset[0].shape)
print("R0 shape:", Rset[0].shape)
print("Number of cameras:", len(Cset), len(Rset))
print("x1Inlier shape:", x1Inlier.shape)
print("x2Inlier shape:", x2Inlier.shape)

# Prepare x2set as a list of Nx2 arrays for other cameras
x2set = [x2Inlier]  # Assuming only one other camera for now, add others as needed

# Initial 3D points from Linear Triangulation
X0 = np.array(Xset[0])[:, 1:4]  # Exclude ID column to get Nx3 array

# Validate shapes of inputs
for i, x2 in enumerate(x2set):
    print(f"x2set[{i}] shape:", x2.shape)
print("X0 shape:", X0.shape)

# Perform Nonlinear Triangulation
X_nl = NonlinearTriangulation(
    K,
    Cset[0],            # First camera center
    Rset[0],            # First camera rotation
    Cset[1:],           # Remaining camera centers
    Rset[1:],           # Remaining camera rotations
    x1Inlier,           # Inlier points in first camera
    x2set,              # Inlier points in other cameras
    X0                  # Initial 3D points
)

# X_nl = np.load("X_nl_set.npy")
print("\nOptimized 3D Points after Nonlinear Triangulation:")
print(X_nl.shape)

################################################################################
# Step 10: PnPRANSAC
# PnPRANSAC: Function to perform PnP using RANSAC to find the best camera pose 
# with inliers
################################################################################
# Prepare 3D points (Xset)
Xset = X_nl[0]  # Extract from (1, 305, 4) to (305, 4)

# Prepare 2D points (xset) with IDs
xset = np.hstack((np.arange(x1Inlier.shape[0]).reshape(-1, 1), x1Inlier))  # Add IDs as the first column

# Validate shapes
print("Xset shape:", Xset.shape)  # Expected (N, 4)
print("xset shape:", xset.shape)  # Expected (  N, 3)

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
# visualize_reprojection(Xset, xset, Cnew, Rnew, K, image_path, output_path)
print("Inliers:", Inliers)
# # Call PnPRANSAC to estimate camera pose
# Cnew, Rnew, Inliers = PnPRANSAC(Xset_df, xset_df, K)
################################################################################
# Step 11: NonLinearPnP
# NonLinearPnP: Refines the camera pose (position and orientation) using non-linear 
# optimization to minimize the reprojection error between observed 2D points and 
# projected 3D points.
################################################################################
X_inliers = Xset[Inliers, :]  # Now X_inliers should be (M,4), M = number of inliers
x_inliers = xset[Inliers, :]  # Similarly x_inliers should be (M,3)

# Make sure the shapes match and correspondences are aligned
print("X_inliers shape:", X_inliers.shape)
print("x_inliers shape:", x_inliers.shape)

# Now call NonlinearPnP with the inlier sets
Copt, Ropt = NonlinearPnP(X_inliers, x_inliers, K, Cnew.reshape(3,1), Rnew)

# Print or use the refined camera pose
print("Refined Camera Center (Copt):", Copt.flatten())
print("Refined Rotation Matrix (Ropt):\n", Ropt)


################################################################################
# Step 12: BuildVisibilityMatrix
# BuildVisibilityMatrix: BuildVisibilityMatrix: Constructs a sparse visibility 
# matrix for bundle adjustment. This matrix indicates which parameters affect 
# each observation in the optimization process.
################################################################################

################################################################################
# Step 13: BundleAdjustment
# BundleAdjustment: Refines camera poses and 3D point positions to minimize the 
# reprojection error for a set of cameras and 3D points using non-linear 
# optimization.
################################################################################
