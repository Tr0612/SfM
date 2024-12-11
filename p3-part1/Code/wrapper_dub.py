from utils import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from GetInliersRANSAC import GetInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from PnPRANSAC import *
from NonlinearTriangulation import NonlinearTriangulation

K = np.array(([568.996140852, 0, 643.21055941], [0,  568.988362396, 477.982801038], [0, 0, 1]))

import os

# Define the file paths for your matching files
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Assuming all previously provided functions are available and imported

# Setup paths for the matching files and images
os.chdir(os.path.dirname(__file__))
print("Current Working Directory:", os.getcwd())

# File paths for matching files
matching_files = [
    '../Data/Imgs/matching1.txt',
    '../Data/Imgs/matching2.txt',
    '../Data/Imgs/matching3.txt',
    '../Data/Imgs/matching4.txt',
    '../Data/Imgs/matching5.txt',
]

# Directory paths for images and outputs
image_dir = "../Data/Imgs/"
output_dir = "../Data/Plots/"
calib_file = "../Data/Imgs/calibration.txt"

# Step 1: Index All Feature Points
if not os.path.exists('../Data/new_matching1.txt'):
    print("\nProcessing Feature Correspondences from matching files...")
    dataframes = IndexAllFeaturePoints(*matching_files)
else:
    print('Refined Features Indexes Already Exists')
    print(bcolors.WARNING + "Warning: Continuing with the existing Feature Indexes..." + bcolors.ENDC)

# Step 2: Load Calibration Matrix
K = process_calib_matrix(calib_file)
print(bcolors.OKCYAN + "\nIntrinsic camera matrix K:" + bcolors.OKCYAN)
print(K, '\n')

# Step 3: Iterate Over All Pairs of Images and Process Matches
for source_idx in range(1, 6):  # Images 1 to 5
    for target_idx in range(source_idx + 1, 7):  # Images 2 to 6
        # Determine the matching file to use based on the source image
        matching_file = matching_files[source_idx - 1]

        print(f"\nProcessing Image Pair: {source_idx} -> {target_idx}")
        keypoints_df = ParseKeypoints(matching_file, sourceIdx=source_idx, targetIdx=target_idx)

        if keypoints_df.empty:
            print(f"No matches found for Image Pair: {source_idx} -> {target_idx}")
            continue

        print(f"Number of Correspondences: {len(keypoints_df)}")

        # Extract source and target keypoints
        try:
            source_keypoints = keypoints_df[[2, 3]].to_numpy()  # x, y for source image
            target_keypoints = keypoints_df[[5, 6]].to_numpy()  # x, y for target image
        except KeyError as e:
            print(f"KeyError for Image Pair: {source_idx} -> {target_idx}")
            print(f"Available Columns: {keypoints_df.columns}")
            raise e

        # Step 4: Apply RANSAC for Outlier Rejection
        # Implement RANSAC logic here or use a provided function
        source_inliers = source_keypoints  # Replace with RANSAC-filtered inliers
        target_inliers = target_keypoints  # Replace with RANSAC-filtered inliers

        # Step 5: Visualize Matches
        DrawMatches(
            imgPath=image_dir,
            sourceIdx=source_idx,
            targetIdx=target_idx,
            sourceAllPts=source_keypoints,
            targetAllPts=target_keypoints,
            sourceInPts=source_inliers,
            targetInPts=target_inliers,
            outputPath=output_dir
        )
# print(indices)
inlier_correspondences  = []
outlier_correspondences = []
for i in range(len(correspondences)):
    if (len(correspondences[i][0])< 4):
        inlier_correspondences.append(correspondences[i])
    else:
        p1 = correspondences[i][0]
        p2 = correspondences[i][1]
        inlier_index, outlier_index  = ransac(p1, p2)
        if i == 0:
            inlier_idx = inlier_index
        inlier_correspondences.append([p1[inlier_index], p2[inlier_index]])
        outlier_correspondences.append([p1[outlier_index], p2[outlier_index]])

for i in range(1,6):
    for j in range(i+1,7):
        features = DrawCorrespondence(i, j, inlier_correspondences[i][0], inlier_correspondences[i][1], outlier_correspondences[i][0], outlier_correspondences[i][0],True)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 1000, 600)
        name = 'feature_correspondence_{}{}.png'.format(i, j)
        cv2.imwrite(name, features)
        # cv2.imshow('image', features)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

total_correspondences = 0
for i in range(0, len(inlier_correspondences)):
    total_correspondences += inlier_correspondences[i][0].shape[0]

visibility_matrix = np.zeros((total_correspondences, 6))

# print(inlier_idx)
F = EstimateFundamentalMatrix(inlier_correspondences[0][0], inlier_correspondences[0][1])
print('********** Fundamental Matrix **********')
pprint(F)
E = compute_essential_matrix(F, K)
print('********** Essential Matrix **********')
pprint(E)
C_set, R_set = estimate_camera_pose(E)
color = ['r', 'g', 'b', 'k']
X_set = []
for i in range(4):
    X = linear_triangulation(K, np.identity(3), R_set[i],  np.zeros((3, 1)),C_set[i].T, inlier_correspondences[0][0],inlier_correspondences[0][1])
    X_set.append(X)
    plt.scatter(X[:, 0], X[:, 2], c=color[i], s=1)
    ax = plt.gca()
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
pprint(np.array(X_set).shape)
plt.savefig('linear_triangulation.png')
plt.close()

R, C , X_linear = find_correct_pose(R_set, C_set, X_set)
print(R.shape)
# print("K shape:", K.shape)
# print("C0 shape:", C.shape)
# print("R0 shape:", inlier_correspondences[0][0].shape)
# print("Number of cameras:", len(Cset), len(Rset))
# print("x1Inlier shape:", X_linear[:, 0:3].shape)
# print("x2Inlier shape:", x2Inlier.shape)
X_nl = non_linear_triangulation(K, np.zeros((3,1)), np.identity(3), C, R,inlier_correspondences[0][0],inlier_correspondences[0][1], X_linear[:, 0:3])
print(f"X_nl {X_nl.shape}")
plt.scatter(X_nl[:, 0], X_nl[:, 2], c=color[i], s=1)
ax = plt.gca()
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.savefig('non-linear.png')
plt.close()

plt.scatter(X_linear[:, 0], X_linear[:, 2], c='r', s=1)
plt.scatter(X_nl[:, 0], X_nl[:, 2], c='k', s=1)
ax = plt.gca()
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.savefig('non-linear_vs_linear.png')
plt.close()

Cset = []
Rset=[]
Cset.append(C)
Rset.append(R)

print('########## Starting PnPRANSAC ##########')
# print(indices[inlier_idx])
for i in range(3, 7):
    p12n, indices12n = find_common_points('Data/matching1.txt', i, indices[inlier_idx])
    # inlier_idx = np.array(inlier_idx)
    inter_indices = np.nonzero(np.in1d(indices, indices12n))[0]
    final_indices = np.nonzero(np.in1d(inlier_idx, inter_indices))[0]
    r_indx = final_indices
    if final_indices.shape[0] != 0:
        x = inlier_correspondences[0][0][final_indices]
        X = X_nl[final_indices]
        print(x.shape)
        print(X.shape)
        Cnew, Rnew = PnPRANSAC(X, x, K)
        Cnew, Rnew = NonLinearPnP(X, x, K, Cnew, Rnew)
        Cset.append(Cnew)
        Rset.append(Rnew)
        Xnew = linear_triangulation(K, np.identity(3), Rnew,  np.zeros((3, 1)),Cnew.T, inlier_correspondences[0][0],inlier_correspondences[0][1])
        Xnew = non_linear_triangulation(K, np.zeros((3,1)), np.identity(3), Cnew , Rnew ,inlier_correspondences[0][0],inlier_correspondences[0][1], Xnew[:, 0:3])
        X_nl = np.vstack((X_nl, Xnew))
        # V_bundle = BuildVisibilityMatrix(visibility_matrix, r_indx)   
        # points = np.hstack((inlier_correspondences[0][0].reshape((-1,1)), inlier_correspondences[0][1].reshape((-1,1))))
        # Rset, Cset, X_3D = BundleAdjustment(Cset, Rset, X_nl, K, points, V_bundle)
    else:
        continue



ax = plt.axes(projection='3d')
ax.scatter3D(X_nl[:, 0], X_nl[:, 1], X_nl[:, 2], s=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim([-0.5, 1])
ax.set_ylim([-0.5, 1])
ax.set_zlim([0, 1.5])

plt.show()
plt.close()

plt.scatter(X_nl[:, 0], X_nl[:, 2], c='r', s=1)
ax = plt.gca()
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.savefig('bundle_adjustment.png')
plt.close()