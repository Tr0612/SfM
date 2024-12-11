import numpy as np

# Function to ensure the rotation matrix R has a positive determinant
def CheckDet(R, C):
    """
    CheckDet: Adjusts the rotation matrix R and translation vector C if the determinant of R is negative.
    This is done to ensure that R represents a valid rotation matrix with a determinant of +1.
    
    Parameters:
    - R: Rotation matrix (3x3).
    - C: Translation vector (3x1).
    
    Returns:
    - Adjusted R and C such that det(R) >= 0.
    """
    # If the determinant of R is -1, invert both R and C
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return R, C


# Function to extract possible camera poses from the Essential Matrix
def ExtractCameraPose(E):
    """
    ExtractCameraPose: Extracts four possible camera poses (rotation and translation pairs) from
    the Essential Matrix (E) using Singular Value Decomposition (SVD).
    
    Parameters:
    - E: Essential matrix (3x3).
    
    Returns:
    - Cset: List of four possible camera translation vectors (3x1).
    - Rset: List of four possible camera rotation matrices (3x3).
    """

    # Perform Singular Value Decomposition (SVD) on the Essential Matrix E
    U, S, Vt = np.linalg.svd(E)


    # Define the rotation matrix W, which is used to construct possible rotations
    W = np.array([[0, -1, 0],
              [1,  0, 0],
              [0,  0, 1]])

    # Compute the four possible camera poses (two rotations and two translations)
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    C1 = U[:, 2]
    C2 = -U[:, 2]

    
    # Ensure each rotation matrix has a positive determinant (R should be a valid rotation matrix)
    R1, C1 = CheckDet(R1, C1)
    R2, C2 = CheckDet(R2, C2)
    R1, C2 = CheckDet(R1, C2)
    R2, C1 = CheckDet(R2, C1)
    
    
    # Expand dimensions of translation vectors for easy concatenation later (make them 3x1)
    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)

    # Collect the four possible camera poses
    Cset = [C1, C2, C1, C2]  # Two translations combined with two rotations
    Rset = [R1, R1, R2, R2]  # Two rotations combined with two translations

    # Display the extracted camera poses with type and shape
    print("\nExtracted Camera Poses:")
    print(f"Type of Cset: {type(Cset)}, Length of Cset: {len(Cset)}")
    print(f"Type of Rset: {type(Rset)}, Length of Rset: {len(Rset)}")

    for i, (C, R) in enumerate(zip(Cset, Rset)):
        print(f"\nPose {i + 1}:")
        print(f"Type of C: {type(C)}, Shape of C: {C.shape}")
        print(f"Type of R: {type(R)}, Shape of R: {R.shape}")
        print("C (Translation Vector):\n", C)
        print("R (Rotation Matrix):\n", R)
        print("Determinant of R:", np.linalg.det(R))

    return Cset, Rset  # Return sets of possible translations and rotations

# ####Test code
# # Helper function to generate a random rotation matrix
# def generate_random_rotation_matrix():
#     """
#     Generates a random 3x3 rotation matrix using the QR decomposition of a random matrix.
#     """
#     random_matrix = np.random.randn(3, 3)
#     Q, R = np.linalg.qr(random_matrix)
#     # Ensure the determinant of Q is +1
#     if np.linalg.det(Q) < 0:
#         Q[:, 2] *= -1
#     return Q

# # Helper function to generate a random translation vector
# def generate_random_translation_vector():
#     """
#     Generates a random 3x1 translation vector.
#     """
#     return np.random.randn(3, 1)

# # Generate a random Essential Matrix (E) for testing
# def generate_essential_matrix():
#     """
#     Generates a random Essential Matrix using a random rotation matrix and translation vector.
#     """
#     R = generate_random_rotation_matrix()
#     t = generate_random_translation_vector()
#     t_skew = np.array([[0, -t[2, 0], t[1, 0]],
#                        [t[2, 0], 0, -t[0, 0]],
#                        [-t[1, 0], t[0, 0], 0]])
#     E = t_skew @ R  # Construct the Essential Matrix
#     return E

# # Generate a random Essential Matrix
# E = generate_essential_matrix()
# print("Generated Essential Matrix (E):\n", E)

# # Test the ExtractCameraPose function
# Cset, Rset = ExtractCameraPose(E)

# # Display the extracted camera poses
# print("\nExtracted Camera Poses:")
# for i, (C, R) in enumerate(zip(Cset, Rset)):
#     print(f"\nPose {i + 1}:")
#     print("C (Translation Vector):\n", C)
#     print("R (Rotation Matrix):\n", R)
#     print("Determinant of R:", np.linalg.det(R))