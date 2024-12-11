import numpy as np

def LinearTriangulation(K, C0, R0, Cseti, Rseti, x1set, x2set):
    """
    LinearTriangulation: Computes 3D points from two sets of 2D correspondences (x1set and x2set)
    observed from two different camera poses using linear triangulation.
    
    Parameters:
    - K: Intrinsic camera matrix (3x3).
    - C0: Camera center for the first camera pose (3x1).
    - R0: Rotation matrix for the first camera pose (3x3).
    - Cseti: Camera center for the second camera pose (3x1).
    - Rseti: Rotation matrix for the second camera pose (3x3).
    - x1set: Array containing 2D points in the first image (ID, u, v).
    - x2set: Array containing corresponding 2D points in the second image (ID, u, v).
    
    Returns:
    - Xset: Array of 3D points in homogeneous coordinates along with their IDs (Nx4).
    """
    Xset = []  # List to store the triangulated 3D points
    
    # Ensure inputs are numpy arrays
    x1set = np.array(x1set)
    x2set = np.array(x2set)
    
    # Validate dimensions
    if x1set.shape[1] != 3 or x2set.shape[1] != 3:
        raise ValueError("Input x1set and x2set must have shape Nx3 (ID, u, v).")

    if x1set.shape[0] != x2set.shape[0]:
        raise ValueError("x1set and x2set must have the same number of correspondences.")

    # Calculate the projection matrices P1 and P2
    I = np.eye(3)  # Identity matrix (3x3)
    P1 = np.matmul(np.matmul(K, R0), np.hstack((I, -C0)))  # Projection matrix for the first camera
    P2 = np.matmul(np.matmul(K, Rseti), np.hstack((I, -Cseti)))  # Projection matrix for the second camera
    
    # Iterate over each pair of corresponding points (x1 in first view and x2 in second view)
    for x1, x2 in zip(x1set, x2set):
        ID = x1[0]  # Unique ID for the point
        u1, v1 = x1[1], x1[2]  # Coordinates in the first image
        u2, v2 = x2[1], x2[2]  # Coordinates in the second image
        
        # Construct matrix A for the linear triangulation system Ax=0
        A = np.zeros((4, 4))
        A[0] = u1 * P1[2] - P1[0]
        A[1] = v1 * P1[2] - P1[1]
        A[2] = u2 * P2[2] - P2[0]
        A[3] = v2 * P2[2] - P2[1]
        
        # Solve Ax=0 using SVD
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]
        
        # Normalize to make the point homogeneous
        X_homogeneous /= X_homogeneous[-1]

        # Append the triangulated 3D point with its ID to the list
        Xset.append([ID, X_homogeneous[0], X_homogeneous[1], X_homogeneous[2]])

    # Convert the list of points to a numpy array for easy manipulation
    Xset = np.array(Xset)

    return Xset  # Return the set of triangulated 3D points with their IDs
