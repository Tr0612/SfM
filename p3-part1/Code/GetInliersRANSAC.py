import pandas as pd
import numpy as np
import random
import sys
import cv2


from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from tqdm import tqdm

def GetInliersRANSAC(x1All, x2All, M=1500, T=0.5, debug=False):
    """
    Estimates the Fundamental matrix using RANSAC and identifies inlier matches
    between two sets of points, rejecting outliers.

    Args:
        x1All (DataFrame): Source image points with IDs and (x, y) coordinates.
        x2All (DataFrame): Target image points with IDs and (x, y) coordinates.
        M (int): Number of RANSAC iterations. Default is 1500.
        T (float): Threshold for inlier selection based on the epipolar constraint. Default is 0.5.

    Returns:
        x1Inlier (DataFrame): Inlier points in the source image.
        x2Inlier (DataFrame): Inlier points in the target image.
        FBest (ndarray): The best estimated Fundamental matrix.
    """

    if debug:
        print("\n------- BEGIN GetInliersRANSAC -------")
        print("---INPUT:--")
        print("Columns in x1All:", x1All.columns)
        print("Columns in x2All:", x2All.columns)
        print(x1All.head())  # Display the first few rows to inspect the data
        print(x2All.head())  # Display the first few rows to inspect the data
        print(f"Parameters - M: {M}, T: {T}")
        print("---EndINPUT:--")

    random_seed = 42
    # Set the random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    best_inlier_count = 0
    FBest = None
    x1Inlier = None
    x2Inlier = None

    # Extract coordinates from dataframes
    x1_coords = x1All[[2, 3]].to_numpy()  # Columns 2 and 3 correspond to x and y
    x2_coords = x2All[[5, 6]].to_numpy()  # Columns 5 and 6 correspond to x and y


    # Debugging: Verify extracted data
    if debug:
        print("Shape of x1_coords:", x1_coords.shape)
        print("Shape of x2_coords:", x2_coords.shape)
        print("First few x1_coords:\n", x1_coords[:5])
        print("First few x2_coords:\n", x2_coords[:5])

    max_idx = len(x1_coords)  # Total number of points

    # RANSAC iterations
    for i in tqdm(range(M)):
        # Step 1: Randomly select 8 pairs of points from the source and target images
        random_indices = random.sample(range(len(x1_coords)), 8)
        # Extract coordinates without IDs for Fundamental matrix estimation
        x1_sample = x1_coords[random_indices]
        x2_sample = x2_coords[random_indices]

        # # Debug: Only log input to EstimateFundamentalMatrix at specific iterations
        if debug:
            if i in {500, 1000, 1500}:
                print(f"--- Debug (Iteration {i}) ---")
                print(f"x1_sample (Type: {type(x1_sample)}, Shape: {x1_sample.shape}):\n{x1_sample}")
                print(f"x2_sample (Type: {type(x2_sample)}, Shape: {x2_sample.shape}):\n{x2_sample}")






        # Step 2: Estimate the Fundamental matrix F from the selected 8-point subsets
            # Call EstimateFundamentalMatrix function here.
        F = EstimateFundamentalMatrix(x1_sample, x2_sample)

            # Debug: Log the estimated Fundamental Matrix
        #if i % (M // 10) == 0:
        #    print(f"Iteration {i}: Estimated Fundamental Matrix (F):\n{F}")

        # Step 3: Check each point pair to see if it satisfies the epipolar constraint
        inlier_indices = []

        for j in range(max_idx): # max_idx: Total number of points
            x1_homogeneous = np.append(x1_coords[j], 1)  # Convert to homogeneous coordinates
            x2_homogeneous = np.append(x2_coords[j], 1)
            # Calculate the epipolar constraint error for the source-target pair
            # Calculate the epipolar constraint error: x2.T * F * x1
            error = abs(np.dot(x2_homogeneous.T, np.dot(F, x1_homogeneous)))

            # Debug: Log the error for the first few points of the first iteration
            #if i == 0 and j < 5:
            #    print(f"Iteration {i}, Point {j}: Epipolar Error: {error}")


            # If the epipolar constraint error is below the threshold T, consider it an inlier
            if error < T:
                inlier_indices.append(j)

                # Debug: Log the number of inliers found in this iteration
        #if i % (M // 10) == 0:
        #    print(f"Iteration {i}: Number of Inliers: {len(inlier_indices)}")

        # Step 4: Update the best Fundamental matrix if the current F has more inliers
        if len(inlier_indices) > best_inlier_count:
            best_inlier_count = len(inlier_indices)
            FBest = F
            x1Inlier = x1All.iloc[inlier_indices]
            x2Inlier = x2All.iloc[inlier_indices]

      # OpenCV implementation for comparison
    F_cv2, inliers_cv2 = cv2.findFundamentalMat(x1_coords, x2_coords, method=cv2.FM_RANSAC, ransacReprojThreshold=T)

    # Filter inliers using the inlier mask from OpenCV
    if inliers_cv2 is not None:
        inlier_indices_cv2 = np.where(inliers_cv2.flatten() == 1)[0]
        x1Inlier_cv2 = x1All.iloc[inlier_indices_cv2]
        x2Inlier_cv2 = x2All.iloc[inlier_indices_cv2]
    else:
        x1Inlier_cv2, x2Inlier_cv2 = pd.DataFrame(), pd.DataFrame()

    geo_thresh = 0.05    
        # Step 4: Final Geometric Filtering
    final_inlier_indices = []
    for i, (x1, x2) in enumerate(zip(x1Inlier[[2, 3]].to_numpy(), x2Inlier[[5, 6]].to_numpy())):
        x1_h = np.append(x1, 1)
        x2_h = np.append(x2, 1)
        distance = abs(np.dot(x2_h.T, np.dot(FBest, x1_h)))
        if distance < geo_thresh:  # Stricter geometric threshold
            final_inlier_indices.append(i)

    x1Inlier = x1Inlier.iloc[final_inlier_indices]
    x2Inlier = x2Inlier.iloc[final_inlier_indices]

    # Print results for comparison
    if debug: 
        print("\n=== Comparison of Fundamental Matrices ===")
        print("Custom Implementation (FBest):\n", FBest)
        print("\nOpenCV Implementation (F_cv2):\n", F_cv2)

        print("\n=== Comparison of Inlier Counts ===")
        print("Custom Implementation Inliers:", len(x1Inlier))
        print("OpenCV Implementation Inliers:", len(x1Inlier_cv2))

    # Return the inlier sets and the best Fundamental matrix

    #return x1Inlier, x2Inlier, FBest
    # return x1Inlier, x2Inlier, FBest
    return x1Inlier_cv2,x2Inlier_cv2,F_cv2




