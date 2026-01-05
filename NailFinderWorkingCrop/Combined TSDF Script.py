import numpy as np
import open3d as o3d
import os
import math
import tkinter as tk 
from tkinter import filedialog 
import copy
import sys

# --- User-Defined Parameters ---
O3D_VISUALIZE = True # Set to True to see all intermediate alignments AND the final model
SERVO_STEP_ANGLE_DEG = 10.0 

# --- Initial Alignment Parameter ---
# ADJUST THIS VALUE based on the visual output to fix the remaining rotation offset.
# Positive values rotate the model clockwise (as viewed from the top, Z-axis).
INITIAL_YAW_ADJUSTMENT_DEG = 0 # <--- ADJUST THIS IF THE FINAL MODEL IS ROTATED
# ---------------------------------------

# --- CRITICAL SCALE CORRECTION ---
# Based on analysis, the point cloud units (1 unit) represent 10 cm (0.1 meters).
# We scale the data to true meters before processing.
POINT_CLOUD_SCALE_FACTOR = 0.1 

# --- Adjusted Z Filters (Now targeting the 6.5 cm distance) ---
# After scaling by 0.1: 0.05m (5cm) to 0.15m (15cm)
Z_FILTER_MIN = 0.01 
Z_FILTER_MAX = 0.15 

# --- Adjusted Parameters for Robustness (kept in true meters) ---
INITIAL_YAW_GUESS_RAD = np.radians(SERVO_STEP_ANGLE_DEG * -1) 
Voxel_Size = 0.002 # 2 mm - THIS IS THE BASE RESOLUTION FOR ALIGNMENT

# --- **TUNED ICP PARAMETERS (Used for high confidence) ---
RANSAC_CORRESPONDENCE_DISTANCE_M = 0.15 
ICP_CORRESPONDENCE_DISTANCE_M = 0.005 
MAX_ICP_ITERATIONS = 400
ICP_THRESHOLD_M = 0.0025
# -------------------------------------------------------------------

# --- ROBUSTNESS PARAMETERS ---
MIN_FITNESS_THRESHOLD = 0.65 
YAW_OUTLIER_SIGMA_THRESHOLD = 1.5 
P_MAGNITUDE_OUTLIER_SIGMA_THRESHOLD = 2.0 
# -----------------------------

# --- MESHING & CLUSTERING PARAMETERS (NEW) ---
# DBSCAN Epsilon: Max distance (in meters) between two points to be considered neighbors.
# Must be smaller than the separation between the nails. (e.g., 5 mm)
DBSCAN_EPS_M = 0.005 
# Minimum points required to form a cluster (filters out small noise fragments)
DBSCAN_MIN_POINTS = 100 

POISSON_DEPTH = 8 
# Output folder for STL files
STL_OUTPUT_DIR = "stl_exports" 

# ======================================================================
# --- File Selection Function (using Tkinter) ---
# ======================================================================

def select_ply_files():
    """Opens a file dialog to select multiple PLY files."""
    try:
        if 'DISPLAY' not in os.environ and os.name != 'nt':
            print("INFO: Tkinter GUI not supported in this environment (headless/non-Windows). Skipping file dialog.")
            return []
        
        root = tk.Tk()
        root.withdraw()
        print("Waiting for file selection dialog...")
        
        file_paths = filedialog.askopenfilenames(
            title="Select PLY Point Cloud Files (in sequence)",
            filetypes=[("PLY files", "*.ply")]
        )
        
        return list(file_paths)
        
    except Exception as e:
        print(f"ERROR: Tkinter file dialog failed ({e}). Reverting to manual list.")
        return []

# ======================================================================
# --- Solver, Helper Functions ---
# ======================================================================

def solve_for_offset_p_v2(R_ideal, t_icp):
    """
    Solves for p_x, p_y (Center of Rotation Offset) assuming p_z = 0,
    based on the equation: t = p - R @ p => (R - I) @ p = -t
    """
    R_xy = R_ideal[:2, :2]
    t_xy = t_icp[:2]
    R_xy_minus_I = R_xy - np.eye(2)
    try:
        p_xy_vector = np.linalg.solve(R_xy_minus_I, -t_xy)
        p_vector = np.array([p_xy_vector[0], p_xy_vector[1], 0.0])
        return p_vector
    except np.linalg.LinAlgError:
        return None
    except Exception:
        return None

def get_rotation_matrix(yaw_rad):
    """Returns a Z-axis rotation matrix (Yaw)."""
    R_yaw = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    return R_yaw

def rotation_matrix_to_euler(R):
    """Converts rotation matrix to Yaw, Pitch, Roll (degrees)."""
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(-R[2, 0])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

def preprocess_point_cloud(pcd, voxel_size):
    """Downsamples and estimates FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    radius_feature = voxel_size * 5 
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh
    
def global_registration(source_down, target_down, source_fpfh, target_fpfh):
    """Initial transformation estimate using RANSAC and FPFH features."""
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, 
        True,
        RANSAC_CORRESPONDENCE_DISTANCE_M, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4, 
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(RANSAC_CORRESPONDENCE_DISTANCE_M)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    
    if result_ransac.inlier_rmse > 0.05:
        print(f" -> WARNING: RANSAC failed (RMSE: {result_ransac.inlier_rmse:.4f}). Falling back to manual guess.")
        initial_guess = np.eye(4)
        initial_guess[:3, :3] = get_rotation_matrix(INITIAL_YAW_GUESS_RAD) 
        return initial_guess
    
    print(f" -> RANSAC Initial Guess Found! RMSE: {result_ransac.inlier_rmse:.4f}")
    return result_ransac.transformation

def get_transform_from_center_of_rotation(p_vector, yaw_rad):
    """
    Constructs the transformation matrix (T) for a rotation (yaw)
    around an off-origin point p.
    """
    R_final = get_rotation_matrix(yaw_rad)
    t_final = p_vector - R_final @ p_vector
    
    T_final = np.eye(4)
    T_final[:3, :3] = R_final
    T_final[:3, 3] = t_final
    
    return T_final
    
def integrate_and_save_mesh(source_pcd, target_pcd, T_icp_step, file_index):
    """
    Transforms, merges, CLUSTERS the point clouds, and saves each cluster
    as a separate STL using Poisson Surface Reconstruction.
    """
    if not os.path.exists(STL_OUTPUT_DIR):
        os.makedirs(STL_OUTPUT_DIR)
        print(f" -> Created output directory: {STL_OUTPUT_DIR}/")

    # 1. Transform the source point cloud and combine
    source_transformed = copy.deepcopy(source_pcd)
    source_transformed.transform(T_icp_step)
    merged_pcd = source_transformed + target_pcd
    
    # 2. Run DBSCAN Clustering to identify separate bodies
    print(f" -> Running DBSCAN (Eps={DBSCAN_EPS_M}m, MinPts={DBSCAN_MIN_POINTS})...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        labels = np.array(merged_pcd.cluster_dbscan(
            eps=DBSCAN_EPS_M, 
            min_points=DBSCAN_MIN_POINTS, 
            print_progress=False
        ))
    
    max_label = labels.max()
    print(f" -> DBSCAN found {max_label + 1} clusters.")
    
    saved_count = 0
    
    # 3. Iterate through each cluster
    for label in range(max_label + 1):
        if label == -1: # Skip noise points
            continue
            
        indices = np.where(labels == label)[0]
        cluster_pcd = merged_pcd.select_by_index(indices)
        
        # Ensure the cluster is large enough to be a real object
        if len(cluster_pcd.points) < DBSCAN_MIN_POINTS:
            continue
            
        # 4. Process a single, isolated cluster
        
        # Estimate/Recompute Normals (CRITICAL for Poisson Reconstruction)
        cluster_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        cluster_pcd.orient_normals_consistent_tangent_plane(100)

        # Poisson Surface Reconstruction
        # Suppress logging for cleaner output
        print(f"    -> Meshing Cluster {label} ({len(cluster_pcd.points)} points)...")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cluster_pcd, 
                depth=POISSON_DEPTH
            )
        
        # Clean up and save the mesh
        mesh.compute_vertex_normals()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        # Save the mesh as STL with cluster ID in the filename
        filename = os.path.join(STL_OUTPUT_DIR, f"alignment_frame_{file_index}_to_{file_index+1}_part_{label}.stl")
        
        if o3d.io.write_triangle_mesh(filename, mesh, write_ascii=False):
            saved_count += 1

    print(f" -> SUCCESS: Saved {saved_count} separate mesh parts for step {file_index}.")
    if saved_count == 0:
        print(" -> WARNING: No clusters were large enough to be saved. Check DBSCAN_EPS_M.")
        return False
    return True

# ======================================================================
# --- The Main Calibration Function (Unchanged Logic) ---
# ======================================================================

def run_calibration():
    
    ply_files = select_ply_files()
    
    if not ply_files or len(ply_files) < 2:
        print("\n--- Manual File Input Fallback ---")
        print("Please manually edit the 'ply_files' list below with the full paths of your PLY scans.")
        ply_files = [
            # >>> USER ACTION REQUIRED: Manually list your PLY file paths here <<<
        ]
        
    if not ply_files or len(ply_files) < 2:
        print("\nFATAL: Need at least 2 file paths for calibration. Please re-run after selecting files or updating the manual list.")
        return

    print(f"Found {len(ply_files)} point clouds. Starting **Robust Calibration**...")
    
    successful_steps = []
    all_cropped_pcds = [] 

    # 1. Parameter Discovery Loop (RANSAC + ICP)
    for i in range(len(ply_files) - 1):
        source_file = ply_files[i]
        target_file = ply_files[i+1]
        
        print(f"\n--- Aligning Frame {i+1} ({os.path.basename(source_file)}) to Frame {i+2} ({os.path.basename(target_file)}) ---")

        source_raw = o3d.io.read_point_cloud(source_file)
        target_raw = o3d.io.read_point_cloud(target_file)
        
        # --- APPLY SCALE CORRECTION (From 10cm units to meters) ---
        source_raw.points = o3d.utility.Vector3dVector(np.asarray(source_raw.points) * POINT_CLOUD_SCALE_FACTOR)
        target_raw.points = o3d.utility.Vector3dVector(np.asarray(target_raw.points) * POINT_CLOUD_SCALE_FACTOR)

        # Apply Z-filter
        source_cropped = source_raw.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-1.0, -1.0, Z_FILTER_MIN), max_bound=(1.0, 1.0, Z_FILTER_MAX)
        ))
        target_cropped = target_raw.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-1.0, -1.0, Z_FILTER_MIN), max_bound=(1.0, 1.0, Z_FILTER_MAX)
        ))
        
        if not source_cropped.has_points() or not target_cropped.has_points():
            print(" -> ERROR: Point cloud is empty after Z-filtering. Check POINT_CLOUD_SCALE_FACTOR and Z_FILTER settings.")
            sys.exit(1) 
            
        if i == 0:
            all_cropped_pcds.append(source_cropped)
        all_cropped_pcds.append(target_cropped)

        source_down, source_fpfh = preprocess_point_cloud(source_cropped, Voxel_Size)
        target_down, target_fpfh = preprocess_point_cloud(target_cropped, Voxel_Size)
            
        T_ransac = global_registration(source_down, target_down, source_fpfh, target_fpfh)

        reg_p2l = o3d.pipelines.registration.registration_icp(
            source_down, target_down, ICP_CORRESPONDENCE_DISTANCE_M, T_ransac, 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ICP_ITERATIONS)
        )
        
        T_icp_step = reg_p2l.transformation
        fitness = reg_p2l.fitness
        
        if fitness < MIN_FITNESS_THRESHOLD:
            print(f" -> WARNING: ICP Alignment Failed (Fitness: {fitness:.4f}). Excluding from average and skipping STL export.")
            skip_parameter_calc = True
            
        else:
            print(f" -> ICP Alignment Successful! Fitness: {fitness:.4f}, RMSE: {reg_p2l.inlier_rmse:.4f}")
            skip_parameter_calc = False

            # --- POISSON Meshing and STL Export (NOW WITH CLUSTERING) ---
            integrate_and_save_mesh(source_cropped, target_cropped, T_icp_step, i+1)

        if O3D_VISUALIZE:
            source_temp = copy.deepcopy(source_down)
            source_temp.transform(T_icp_step)
            source_temp.paint_uniform_color([0, 0, 1])
            target_down.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([source_temp, target_down], 
                                 window_name=f"Frame {i+1} (Blue) aligned to Frame {i+2} (Red) - Fitness: {fitness:.4f}")

        R_icp_step = T_icp_step[:3, :3]
        t_icp_step = T_icp_step[:3, 3]
        observed_yaw_deg, observed_pitch_deg, observed_roll_deg = rotation_matrix_to_euler(R_icp_step)
        print(f" -> Observed Step Rotation: Yaw={observed_yaw_deg:.4f} deg, Pitch={observed_pitch_deg:.4f} deg, Roll={observed_roll_deg:.4f} deg")
        
        ideal_yaw_rad_observed = np.radians(observed_yaw_deg)
        R_ideal_observed_yaw = get_rotation_matrix(ideal_yaw_rad_observed) 

        p_vector = solve_for_offset_p_v2(R_ideal_observed_yaw, t_icp_step)
        
        if p_vector is not None: 
            if not skip_parameter_calc:
                successful_steps.append({
                    'p': p_vector, 
                    'yaw': observed_yaw_deg, 
                    'roll': observed_roll_deg, 
                    'pitch': observed_pitch_deg,
                    'file_pair': f"Frame {i+1} to {i+2}"
                })
            
            print(f" -> Discovered Center of Rotation Offset p (m): [{p_vector[0]:.5f} {p_vector[1]:.5f} {p_vector[2]:.5f}]")
        
    # 5. Final Parameter Averaging and Output
    if not successful_steps:
        print("\n--- Calibration Failed ---")
        print(f"No ICP alignments achieved the minimum fitness threshold of {MIN_FITNESS_THRESHOLD}. Review parameters and data quality.")
        return

    all_yaws = np.array([step['yaw'] for step in successful_steps])
    all_p_mags = np.array([np.linalg.norm(step['p']) for step in successful_steps])
    
    mean_yaw = np.mean(all_yaws)
    std_yaw = np.std(all_yaws)
    
    mean_p_mag = np.mean(all_p_mags)
    std_p_mag = np.std(all_p_mags)

    print(f"\n--- Robust Calibration Filter (Mean Yaw: {mean_yaw:.4f} deg, Std Dev: {std_yaw:.4f} deg) ---")
    print(f"--- P-Vector Mag Filter (Mean |p|: {mean_p_mag:.4f} m, Std Dev: {std_p_mag:.4f} m) ---")
    
    robust_steps = []
    
    for step in successful_steps:
        is_yaw_outlier = abs(step['yaw'] - mean_yaw) > YAW_OUTLIER_SIGMA_THRESHOLD * std_yaw
        p_mag = np.linalg.norm(step['p'])
        is_p_mag_outlier = abs(p_mag - mean_p_mag) > P_MAGNITUDE_OUTLIER_SIGMA_THRESHOLD * std_p_mag
        
        if is_yaw_outlier or is_p_mag_outlier:
            print(f" -> Excluding {step['file_pair']} (Yaw: {step['yaw']:.4f} deg, |p|: {p_mag:.4f} m) as outlier.")
        else:
            robust_steps.append(step)

    if not robust_steps:
        print("FATAL: All successful steps were classified as outliers. Cannot calculate a robust average.")
        robust_steps = successful_steps
        print("Reverting to simple average of all successful steps.")

    final_p_vectors = np.array([step['p'] for step in robust_steps])
    final_rolls = np.array([step['roll'] for step in robust_steps])
    final_pitches = np.array([step['pitch'] for step in robust_steps])
    final_yaws = np.array([step['yaw'] for step in robust_steps])

    avg_p = np.mean(final_p_vectors, axis=0)
    avg_roll = np.mean(final_rolls)
    avg_pitch = np.mean(final_pitches)
    avg_yaw_step = np.mean(final_yaws)
    yaw_correction_factor = avg_yaw_step / SERVO_STEP_ANGLE_DEG
    
    print(f" -> Final robust average calculated from {len(robust_steps)} steps.")

    # 6. Final Combined Visualization using CALIBRATED PARAMETERS
    print("\nAttempting final combined point cloud visualization using CALIBRATED PARAMETERS...")
    
    R_wobble_avg = np.dot(
        o3d.geometry.get_rotation_matrix_from_xyz((0, np.radians(avg_pitch), 0)),
        o3d.geometry.get_rotation_matrix_from_xyz((np.radians(avg_roll), 0, 0))
    )
    
    R_wobble_inverse = R_wobble_avg.T
    T_wobble_inverse = np.eye(4)
    T_wobble_inverse[:3, :3] = R_wobble_inverse
    
    step_yaw_rad = np.radians(SERVO_STEP_ANGLE_DEG * yaw_correction_factor * -1) 
    
    T_step_calibrated_pure = get_transform_from_center_of_rotation(
        avg_p, 
        step_yaw_rad
    )
    
    combined_pcd = o3d.geometry.PointCloud()
    cumulative_calibrated_T = np.eye(4)
    
    for i, pcd_raw in enumerate(all_cropped_pcds):
        
        pcd = copy.deepcopy(pcd_raw) 

        pcd.transform(T_wobble_inverse)

        if i == 0:
            R_yaw_adj = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(INITIAL_YAW_ADJUSTMENT_DEG)))
            T_yaw_adj = np.eye(4)
            T_yaw_adj[:3, :3] = R_yaw_adj
            pcd.transform(T_yaw_adj)
            
        pcd.transform(cumulative_calibrated_T)
        
        combined_pcd += pcd
        
        cumulative_calibrated_T = cumulative_calibrated_T @ T_step_calibrated_pure
        
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=Voxel_Size * 1.0) 
    
    o3d.visualization.draw_geometries([combined_pcd], window_name="Final Combined Calibrated Point Cloud (Drift Corrected)")
    
    # 7. Print Results
    print("\n" + "="*50)
    print("      ✅ CALIBRATION COMPLETE: PASTE VALUES INTO LIVE SCRIPT   ")
    print("="*50)
    
    print(f"\nNOTE: Separate STL files for each identified cluster were saved in the '{STL_OUTPUT_DIR}/' directory.")
    
    print("\n--- 1. Center of Rotation Offset (CALIBRATED_OFFSET_P) ---")
    print(f"    X Offset: {avg_p[0]:.5f} m")
    print(f"    Y Offset: {avg_p[1]:.5f} m")
    print(f"    Z Offset: {avg_p[2]:.5f} m")
    print(f"    COPY LINE: np.array([{avg_p[0]:.5f}, {avg_p[1]:.5f}, {avg_p[2]:.5f}])") 
    
    print("\n--- 2. Static Camera Wobble (CALIBRATED_ROLL/PITCH_DEG) ---")
    print(f"    Roll (X-wobble): {avg_roll:.4f} deg")
    print(f"    Pitch (Y-wobble): {avg_pitch:.4f} deg")
    
    print("\n--- 3. Servo Step Correction (YAW_CORRECTION_FACTOR) ---")
    print(f"    Average True Step Yaw: {avg_yaw_step:.4f} deg")
    print(f"    Correction Factor: {yaw_correction_factor:.5f}")
    
    print("\n--- 4. Initial Yaw Alignment (Aesthetic Correction) ---")
    print(f"    Current Adjustment: {INITIAL_YAW_ADJUSTMENT_DEG:.1f} deg")
    print("     ACTION: Adjust 'INITIAL_YAW_ADJUSTMENT_DEG' in the script if the final model needs rotation in the XY plane.")
    
    print("="*50)

# --- Add Main Execution Block ---
if __name__ == "__main__":
    run_calibration()