from scipy.spatial.transform import Rotation
import numpy as np
from scipy.optimize import least_squares

def BundleAdjustment(Call, Rall, Xall, K, sparseVmatrix, n_cameras, n_points, camera_indices, point_indices, xall):
    """
    Refines camera poses and 3D point positions to minimize the reprojection error.

    Parameters:
    - Call: Initial camera positions (list of 3x1 arrays).
    - Rall: Initial rotation matrices for each camera (list of 3x3 arrays).
    - Xall: Initial 3D points (Nx3).
    - K: Intrinsic camera matrix (3x3).
    - sparseVmatrix: Sparse visibility matrix.
    - n_cameras: Number of cameras.
    - n_points: Number of 3D points.
    - camera_indices: Indices of cameras observing each point.
    - point_indices: Indices of 3D points corresponding to observations.
    - xall: Observed 2D points (Nx2).

    Returns:
    - CoptAll: Optimized camera positions (list of 3x1 arrays).
    - RoptAll: Optimized rotation matrices (list of 3x3 arrays).
    - XoptAll: Optimized 3D points (Nx3).
    """
    def reprojection_loss(params, n_cameras, n_points, camera_indices, point_indices, xall, K):
        """
        Computes the reprojection error for all cameras and points.
        """
        # Extract camera parameters
        camera_params = params[:n_cameras * 7].reshape((n_cameras, 7))
        points_3D = params[n_cameras * 7:].reshape((-1, 3))  # Match number of points

        # Compute projection matrices and errors
        residuals = []
        for i in range(len(xall)):
            cam_idx = camera_indices[i]
            pt_idx = point_indices[i]

            # Camera parameters
            t = camera_params[cam_idx, :3].reshape(3, 1)
            q = camera_params[cam_idx, 3:]
            R = Rotation.from_quat(q).as_matrix()

            # Projection matrix
            P = np.dot(K, np.hstack((R, -R @ t)))

            # 3D point in homogeneous coordinates
            X_hom = np.append(points_3D[pt_idx], 1)

            # Project to image
            proj = P @ X_hom
            proj /= proj[2]  # Normalize

            # Compute reprojection error
            residuals.extend((xall[i] - proj[:2]).ravel())

        return np.array(residuals)

    # Debug input sizes
    print(f"Number of Cameras: {n_cameras}, Camera Params Size: {n_cameras * 7}")
    print(f"Number of Points: {n_points}, Point Params Size: {n_points * 3}")
    print(f"Xall Shape: {Xall.shape}, Expected Points: {n_points}")

    # Prepare initial parameter vector
    camera_params = []
    for C, R in zip(Call, Rall):
        q = Rotation.from_matrix(R).as_quat()
        camera_params.append(np.hstack((C.ravel(), q)))
    camera_params = np.array(camera_params).ravel()

    # Ensure Xall matches expected number of points
    if Xall.shape[0] != n_points:
        raise ValueError(f"Mismatch: Xall has {Xall.shape[0]} points, but expected {n_points}")

    points_3D = Xall.ravel()
    init_params = np.hstack((camera_params, points_3D))

    # Validate total parameter length
    expected_param_length = n_cameras * 7 + n_points * 3
    if len(init_params) != expected_param_length:
        raise ValueError(f"Mismatch: Expected {expected_param_length} params, but got {len(init_params)}")

    # Run optimization
    result = least_squares(
        reprojection_loss,
        init_params,
        jac_sparsity=sparseVmatrix,
        method='trf',
        verbose=2,
        x_scale='jac',
        loss='linear',
        ftol=1e-4,
        xtol=1e-4,
        args=(n_cameras, n_points, camera_indices, point_indices, xall, K)
    )

    # Extract optimized parameters
    optimized_params = result.x

    # Extract optimized camera positions and rotations
    optimized_camera_params = optimized_params[:n_cameras * 7].reshape((n_cameras, 7))
    CoptAll = [optimized_camera_params[i, :3].reshape(3, 1) for i in range(n_cameras)]
    RoptAll = [Rotation.from_quat(optimized_camera_params[i, 3:]).as_matrix() for i in range(n_cameras)]

    # Extract optimized 3D points
    XoptAll = optimized_params[n_cameras * 7:].reshape((-1, 3))

    return CoptAll, RoptAll, XoptAll
