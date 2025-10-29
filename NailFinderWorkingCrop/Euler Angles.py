import numpy as np
import open3d as o3d

# Your successful ICP Rotation Matrix (3x3 block)
R_icp = np.array([
    [0.9392133, 0.3427594, -0.01985856],
    [-0.34328177, 0.93850848, -0.03687072],
    [ 0.00599964, 0.04144655, 0.99912271]
])

# Convert R to Euler angles (XYZ: Roll, Pitch, Yaw)
# Open3D's function is get_rotation_matrix_from_xyz()
# We use numpy math to reverse the process:
def R_to_euler_xyz(R):
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees(np.array([roll, pitch, yaw]))

# Calculate the rotation inherent in the 10-degree step
euler_deg = R_to_euler_xyz(R_icp)

# Roll (X-axis), Pitch (Y-axis), Yaw (Z-axis)
AVG_ROLL_PITCH = euler_deg[:2]  # Roll and Pitch (X, Y)
print("\nCalculated Roll and Pitch (Degrees):")
print(f"Roll (X): {AVG_ROLL_PITCH[0]:.4f}")
print(f"Pitch (Y): {AVG_ROLL_PITCH[1]:.4f}")
print(f"Yaw (Z): {euler_deg[2]:.4f} (Should be near 10 degrees)")