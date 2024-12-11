import numpy as np
from scipy.optimize import least_squares

# Function to perform Non-linear Triangulation to refine 3D points given initial estimates
def NonlinearTriangulation(K, C0, R0, Cset, Rset, x1set, x2set, X0):
    """
    Optimizes the 3D points by minimizing the reprojection error using Nonlinear Triangulation.
    
    Parameters:
    - K: Intrinsic camera matrix (3x3).
    - C0: Translation vector of the first camera (3x1).
    - R0: Rotation matrix of the first camera (3x3).
    - Cset: List of translation vectors of other cameras (each 3x1 array).
    - Rset: List of rotation matrices of other cameras (each 3x3 array).
    - x1set: 2D points in the first camera image (Nx2 array).
    - x2set: 2D points in the second camera images (list of Nx2 arrays).
    - X0: Initial 3D points for optimization (Nx3 array).
    
    Returns:
    - X_optimized: Optimized 3D points after minimizing reprojection error (Nx3 array).
    """
    
    def reprojection_error(X_flat, K, C0, R0, Cset, Rset, x1set, x2set):
        """
        Computes the reprojection error for all cameras and points.
        """
        X = X_flat.reshape(-1, 3)  # Reshape flattened X to (Nx3)
        errors = []

        # First camera projection matrix
        P0 = K @ R0 @ np.hstack((np.eye(3), -C0))

        for i, X_world in enumerate(X):
            # Homogeneous coordinates for the 3D point
            X_hom = np.append(X_world, 1)
            
            # Project point onto the first camera
            proj1 = P0 @ X_hom
            proj1 /= proj1[2]  # Normalize to (u, v)
            reproj1 = proj1[:2]
            
            # Compute error for the first camera
            error1 = x1set[i] - reproj1
            errors.extend(error1)
            
            # For each additional camera
            for Cj, Rj, x2 in zip(Cset, Rset, x2set):
                # Ensure Cj has shape (3, 1)
                Cj = np.array(Cj).reshape(3, 1)
                Pj = K @ Rj @ np.hstack((np.eye(3), -Cj))
                
                # Project point onto the current camera
                projj = Pj @ X_hom
                projj /= projj[2]  # Normalize to (u, v)
                reprojj = projj[:2]
                
                # Compute error for the current camera
                errorj = x2[i] - reprojj
                errors.extend(errorj)

        return np.array(errors)
    
    # Flatten initial 3D points for optimization
    X0_flat = X0.reshape(-1)

    # Perform optimization
    result = least_squares(
        reprojection_error,
        X0_flat,
        method='lm',
        max_nfev=100000,
        verbose = 2,
        # bounds=(-1e6, 1e6),
        args=(K, C0, R0, Cset, Rset, x1set, x2set)
    )

    # Reshape the optimized 3D points to (Nx3)
    X_optimized = result.x.reshape(-1, 3)

    return X_optimized