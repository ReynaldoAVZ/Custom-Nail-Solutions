import open3d as o3d
import numpy as np
import math
import os
import sys
from PyQt6 import QtWidgets

# =========================================================================
# === CONFIGURATION SECTION (CRITICAL: ENTER YOUR CALCULATED VALUES) ======
# =========================================================================

# 1. CORRECTED OFFSET (p) for the Center of Rotation
# This is the vector calculated from the 2x2 linear solver:
CORRECTED_OFFSET = np.array([-0.05872277, 0.0439813, 0.0])

# 2. FIXED MECHANICAL TILT CORRECTION (Roll/Pitch)
# These values compensate for the servo not being perfectly vertical (X/Y wobble).
FIXED_ROLL_DEG = 2.3754  # <-- YOUR CALCULATED ROLL (X-axis)
FIXED_PITCH_DEG = -0.3438 # <-- YOUR CALCULATED PITCH (Y-axis)

# 3. DECLARE ANGLES FOR EACH CAPTURED FRAME
# IMPORTANT: This list MUST match the angle and order of the .ply files you select.
FRAME_ANGLES_DEG = [60, 70, 80]

# ICP Parameters (For alignment refinement)
ICP_THRESHOLD = 0.015 # 15 mm maximum distance
ICP_MAX_ITER = 100    # More iterations for better convergence

# Visualization/TSDF Parameters
VOXEL_SIZE = 0.002 # 2 mm for downsampling/volume

# =========================================================================
# === HELPER FUNCTIONS ====================================================
# =========================================================================

def get_files_dialog():
    """Opens a file selector dialog to choose multiple .ply files."""
    # Ensure a QApplication exists for the dialog
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
        
    file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
        None, 
        "Select ALL individual PLY frames in sequential order", 
        os.path.expanduser('~'), 
        "Point Cloud Files (*.ply)"
    )
    # The QApplication must be explicitly quit or it will hang sometimes
    if not QtWidgets.QApplication.instance().parent():
        app.quit()
    return file_names

def calculate_extrinsic(angle_deg, offset, fixed_roll_deg, fixed_pitch_deg):
    """
    Calculates the 4x4 extrinsic camera pose matrix, incorporating 
    the corrected offset AND the fixed mechanical roll/pitch.
    """
    # Convert all angles to radians
    yaw_rad = math.radians(angle_deg)
    inverted_yaw_rad = math.radians(-angle_deg)  # Invert for correct direction
    roll_rad = math.radians(fixed_roll_deg)
    pitch_rad = math.radians(fixed_pitch_deg)
    
    # 1. Fixed Roll/Pitch (X/Y) Rotation Matrix
    # We create a matrix for the fixed tilt.
    R_roll_pitch = o3d.geometry.get_rotation_matrix_from_xyz((roll_rad, pitch_rad, 0.0))
    
    # 2. Variable Yaw (Z) Rotation Matrix
    R_yaw = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, inverted_yaw_rad))
    
    # 3. Total Rotation: Yaw is applied *on top of* the fixed Roll/Pitch.
    R_total = R_yaw @ R_roll_pitch 
    
    # 4. Build the full extrinsic transformation: T_pos @ R_total @ T_neg
    T_neg = np.eye(4); T_neg[:3, 3] = -offset
    T_rotation = np.eye(4); T_rotation[:3, :3] = R_total
    T_pos = np.eye(4); T_pos[:3, 3] = offset
    
    extrinsic = T_pos @ T_rotation @ T_neg
    return extrinsic

def get_unique_color(index, total):
    """Generates a unique color based on the frame index."""
    hue = index / total
    r = (math.sin(hue * 2 * math.pi + 0) * 0.5 + 0.5)
    g = (math.sin(hue * 2 * math.pi + 2) * 0.5 + 0.5)
    b = (math.sin(hue * 2 * math.pi + 4) * 0.5 + 0.5)
    return [r, g, b]

# =========================================================================
# === MAIN EXECUTION ======================================================
# =========================================================================

def run_fusion_and_visualization_with_refinement():
    print("--- Aligned Frame Visualizer with Full Target Alignment ---")
    
    file_paths = get_files_dialog()
    if not file_paths or len(file_paths) != len(FRAME_ANGLES_DEG):
        print("ERROR: File count mismatch or no files selected. Exiting.")
        return

    num_frames = len(file_paths)

    # --- 1. Setup the FIXED Target Frame (Frame 1) ---
    try:
        # Load the original point cloud for Frame 1
        target_pcd_orig = o3d.io.read_point_cloud(file_paths[0])
        
        # Apply the CORRECT extrinsic to place it in the 'correct' world pose
        target_extrinsic = calculate_extrinsic(
            FRAME_ANGLES_DEG[0], CORRECTED_OFFSET, FIXED_ROLL_DEG, FIXED_PITCH_DEG
        )
        target_pcd = o3d.geometry.PointCloud(target_pcd_orig).transform(target_extrinsic)
        
        target_pcd.paint_uniform_color(get_unique_color(0, num_frames))
        visual_pcds = [target_pcd]
        print(f"Loaded Frame 1 ({FRAME_ANGLES_DEG[0]} deg) as FIXED TARGET.")
        
    except Exception as e:
        print(f"Error loading initial target file: {e}")
        return

    # --- 2. Iteratively align all subsequent frames to the FIXED Target Frame ---
    
    for i in range(1, num_frames):
        current_angle = FRAME_ANGLES_DEG[i]
        
        try:
            # Load the source point cloud
            source_pcd = o3d.io.read_point_cloud(file_paths[i])

            # Calculate the CORRECT extrinsic for the current frame
            T_source_correct = calculate_extrinsic(
                current_angle, CORRECTED_OFFSET, FIXED_ROLL_DEG, FIXED_PITCH_DEG
            )
            
            # Use the correct extrinsic to transform the point cloud into its IDEAL world position
            source_pcd.transform(T_source_correct)
            
            # --- ICP REFINEMENT: Snap the current frame to the FIXED target ---
            # ICP finds the small correction needed due to acquisition error/initial bad save.
            
            source_down = source_pcd.voxel_down_sample(VOXEL_SIZE)
            target_down = target_pcd.voxel_down_sample(VOXEL_SIZE)

            print(f"  Aligning Frame {i+1} ({current_angle} deg) to FIXED TARGET...")
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_down, target_down, ICP_THRESHOLD, np.eye(4), 
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER)
            )
            
            # Apply the final ICP transformation to the full-resolution source cloud
            source_pcd.transform(reg_p2p.transformation)
            
            # Color and add to visualization list
            color = get_unique_color(i, num_frames)
            source_pcd.paint_uniform_color(color)
            visual_pcds.append(source_pcd)
            
            # Merge the newly aligned cloud into the target for the next iteration
            target_pcd += source_pcd
            target_pcd = target_pcd.voxel_down_sample(VOXEL_SIZE)

            print(f"    Fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}")
            
        except Exception as e:
            print(f"Error processing frame {i+1}: {e}")

    # --- 3. Visualization ---
    
    if visual_pcds:
        final_pcd = o3d.geometry.PointCloud()
        for pcd in visual_pcds:
            final_pcd += pcd.voxel_down_sample(VOXEL_SIZE)

        print("\nDisplaying final point cloud. Check for smooth transitions between colors.")
        o3d.visualization.draw_geometries(
            [final_pcd], 
            window_name="FINAL: Fully Corrected and Aligned Point Cloud",
            width=1024, height=768
        )
    else:
        print("No point clouds were successfully loaded for visualization.")

# Execute the main function
if __name__ == "__main__":
    run_fusion_and_visualization_with_refinement()