import numpy as np

# --- 1. Data from your successful ICP run ---
t_icp = np.array([0.01152724, 0.02282345, 0.00175603])
tx, ty, tz = t_icp

# The Rotation Angle (theta) is extracted from the R11 component:
# R11 = 0.9392133 --> theta = 20.07 degrees (0.3503 radians)
theta = 0.3503  # in radians

# --- 2. Solve for p_x and p_y (The Critical Components) ---
c, s = np.cos(theta), np.sin(theta)

# M_2x2 is the 2x2 matrix from the 2D rotation-offset formula
M_2x2 = np.array([
    [ 1-c,  s ],
    [-s,  1-c ]
])

t_2x1 = np.array([tx, ty])

try:
    # Use numpy's solver, which is robust for this system
    p_xy = np.linalg.solve(M_2x2, t_2x1)
    p_x, p_y = p_xy

    # --- 3. Determine p_z ---
    # Since rotation is around Z, the p_z component is NOT determined by the rotation formula.
    # It must be set based on the non-zero Z-translation (tz) observed.
    # A safe, conservative choice is to assume the Z-translation is a result of the
    # camera's vertical position and use the non-zero tz as a guide, or assume it's small.
    # Given the high confidence in the overall setup:
    # We will assume that the Z-component of the offset is simply 0, 
    # since any Z-offset would NOT cause a Z-translation during a perfect Z-rotation.
    # The non-zero tz must be due to the slight roll/pitch, which we're ignoring for p.
    p_z = 0.0 # Set p_z to 0.0 meters for a pure Z-rotation model

    p_offset = np.array([p_x, p_y, p_z])

    print("\n--- Final, Corrected Offset Vector p (meters) ---")
    print(p_offset)
    print("-------------------------------------------------")
    
except np.linalg.LinAlgError:
    print("Error: Could not solve the 2x2 system. The rotation angle is too small.")