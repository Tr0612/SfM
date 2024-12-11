import numpy as np
from scipy.sparse import lil_matrix

# Function to create a visibility matrix from inlier data
def GetVmatrix(All_Inlier):
    """
    GetVmatrix: Extracts and combines u, v coordinates, Point IDs, and Camera IDs from inlier data.

    Parameters:
    - All_Inlier: DataFrame or array where each row contains PointID, (u, v) coordinates, and CameraID.

    Returns:
    - Vmatrix: Combined matrix of u, v coordinates, Point IDs, and Camera IDs.
    """
    # Extract components from the array
    point_ids = All_Inlier[:, 0]  # Point IDs
    u_coords = All_Inlier[:, 1]  # u-coordinates
    v_coords = All_Inlier[:, 2]  # v-coordinates
    camera_ids = All_Inlier[:, 3]  # Camera IDs

    # Concatenate u, v, PointID, and CameraID into a single matrix
    Vmatrix = np.column_stack((point_ids, u_coords, v_coords, camera_ids))
    
    return Vmatrix

# Function to build the visibility matrix for bundle adjustment
def BuildVisibilityMatrix(n_cameras, n_points, camera_indices, point_indices):
    """
    BuildVisibilityMatrix: Constructs a sparse visibility matrix for bundle adjustment.
    This matrix indicates which parameters affect each observation in the optimization process.

    Parameters:
    - n_cameras: Total number of cameras.
    - n_points: Total number of 3D points.
    - camera_indices: Array of indices indicating which camera observes each 2D point.
    - point_indices: Array of indices indicating which 3D point corresponds to each 2D point.

    Returns:
    - A: Sparse visibility matrix in lil_matrix format.
    """
    # Calculate the number of observations (2D points), each observation contributes two rows (u and v)
    n_observations = len(camera_indices)

    # Calculate the number of parameters (unknowns) in the optimization
    # Each camera has 7 parameters (3 for translation, 4 for rotation as quaternion)
    # Each 3D point has 3 parameters (X, Y, Z)
    n_camera_params = 7  # Parameters per camera (3 translation + 4 quaternion rotation)
    n_point_params = 3   # Parameters per 3D point (X, Y, Z)

    n_total_params = n_cameras * n_camera_params + n_points * n_point_params

    # Initialize a sparse matrix in 'list of lists' (lil) format for efficient row operations
    A = lil_matrix((n_observations * 2, n_total_params), dtype=int)

    # Create an array of observation indices
    obs_indices = np.arange(n_observations)

    # Fill in the visibility matrix for camera parameters
    for obs_idx, cam_idx in enumerate(camera_indices):
        camera_start = cam_idx * n_camera_params
        A[2 * obs_idx, camera_start:camera_start + n_camera_params] = 1  # u-coordinate
        A[2 * obs_idx + 1, camera_start:camera_start + n_camera_params] = 1  # v-coordinate

    # Fill in the visibility matrix for 3D point parameters
    for obs_idx, point_idx in enumerate(point_indices):
        point_start = n_cameras * n_camera_params + point_idx * n_point_params
        A[2 * obs_idx, point_start:point_start + n_point_params] = 1  # u-coordinate
        A[2 * obs_idx + 1, point_start:point_start + n_point_params] = 1  # v-coordinate

    return A  # Return the sparse visibility matrix

################################################################################
# Step 12: BuildVisibilityMatrix
################################################################################
# Example Usage
# n_cameras = 5
# n_points = 10
# camera_indices = np.random.randint(0, n_cameras, 20)  # Random camera indices
# point_indices = np.random.randint(0, n_points, 20)    # Random point indices

# # Build the visibility matrix
# visibility_matrix = BuildVisibilityMatrix(n_cameras, n_points, camera_indices, point_indices)

# # Print the visibility matrix shape and a sample
# print("Visibility Matrix Shape:", visibility_matrix.shape)
# print("Sample Visibility Matrix:", visibility_matrix.toarray())
