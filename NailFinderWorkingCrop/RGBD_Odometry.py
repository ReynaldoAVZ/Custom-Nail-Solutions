import open3d as o3d
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog
import json 
import copy 
import os 
import time

# Attempt to import matplotlib for visualization, exit if unavailable
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[FATAL ERROR] Matplotlib is required for colored visualization. Please install it (pip install matplotlib).")
    sys.exit(1)

# Set the registration pipeline module for clarity
o3dr = o3d.pipelines.registration

# --- Helper Function for File Selection ---

def select_file(title, filetypes):
    """Opens a Tkinter file dialog to select a single file."""
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if not filepath:
        print(f"\n[ERROR] File selection cancelled or failed for: {title}")
        sys.exit(1)
    return filepath

def select_multiple_files(title, filetypes):
    """
    Opens a Tkinter file dialog to select multiple files. 
    NOTE: Files must be selected in sequential order (Frame 0, Frame 1, Frame 2, ...).
    """
    root = tk.Tk()
    root.withdraw()
    filepaths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    
    if not filepaths:
        print(f"\n[ERROR] Multi-file selection cancelled or failed for: {title}")
        sys.exit(1)
        
    print(f"Selected {len(filepaths)} files for: {title.split('(')[0].strip()}")
    return list(filepaths)

# --- Dedicated Visualization Helper ---
def visualize_geometry(geometries, window_name="Open3D Visualization"):
    """Creates a new temporary visualization window for the given geometries."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768)
    
    # Ensure all geometries are in a list for iteration
    if not isinstance(geometries, list):
        geometries = [geometries]

    for geo in geometries:
        vis.add_geometry(geo)

    ctr = vis.get_view_control()
    # Basic preset view
    ctr.set_zoom(0.8)
    ctr.set_front([0.0, -0.2, -0.98])
    ctr.set_lookat([0, 0, 1.5])
    ctr.set_up([0, -0.98, 0.2])

    print(f"\n[VISUALIZE] Showing: {window_name}. Close window to proceed.")
    vis.run()
    vis.destroy_window()

# --- ICP Registration Function ---
def refine_with_icp(source_pcd, target_pcd, initial_transform, max_corr_dist=0.1, max_iterations=2000):
    """
    Performs Point-to-Plane ICP registration.
    """
    threshold = max_corr_dist
    
    criteria = o3dr.ICPConvergenceCriteria(
        max_iteration=max_iterations, 
        relative_fitness=1e-6, 
        relative_rmse=1e-6
    )

    # Note: Using PointToPlane is generally better for surface reconstruction
    reg_result = o3dr.registration_icp(
        source_pcd, target_pcd, threshold, initial_transform,
        o3dr.TransformationEstimationPointToPlane(),
        criteria
    )
    
    return reg_result

def preprocess_for_icp(pcd, voxel_size, do_outlier_removal=True):
    """
    Downsample, clean, and orient normals to stabilize ICP, especially on curved surfaces.
    """
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if do_outlier_removal:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)
    return pcd

def multi_scale_icp(source, target, initial_transform, voxel_scales, max_corr_mult=4.0):
    """
    Runs coarse-to-fine ICP; each stage feeds the next transform.
    """
    current_transform = initial_transform
    for voxel_size in voxel_scales:
        src_ds = preprocess_for_icp(source, voxel_size)
        tgt_ds = preprocess_for_icp(target, voxel_size)
        reg = refine_with_icp(
            source_pcd=src_ds,
            target_pcd=tgt_ds,
            initial_transform=current_transform,
            max_corr_dist=voxel_size * max_corr_mult,
            max_iterations=200
        )
        current_transform = reg.transformation
    return current_transform, reg

# --- Main Script Flow ---
def run_odometry_registration():
    """
    Executes the multi-frame point cloud registration pipeline using RGBD Odometry 
    for initial estimation, followed by ICP for refinement.
    """
    print("--- Multi-Frame Odometry + ICP Registration Pipeline ---")
    
    # Define accepted file types
    pcd_filetypes = [("Point Cloud files", "*.pcd *.ply")]
    img_filetypes = [("Image files", "*.png *.jpg *.jpeg")]

    try:
        # 1. Intrinsic and File Selection
        intrinsic_path = select_file("Select Camera Intrinsic JSON File (fx/fy/cx/cy format)", [("JSON files", "*.json")])
        
        print("\n--- 1. Select all sequential frame files (Frame 0, 1, 2, ... N) ---")
        color_paths = select_multiple_files("Select N Color Images (in order)", img_filetypes)
        depth_paths = select_multiple_files("Select N Depth Images (in order)", [("PNG files", "*.png")])
        pcd_paths = select_multiple_files("Select N Point Cloud Files (in order)", pcd_filetypes)
        mask_pcd_paths = select_multiple_files("Select N Mask Point Cloud Files (in order)", pcd_filetypes)

        N = len(color_paths)
        if not (len(depth_paths) == N and len(pcd_paths) == N and len(mask_pcd_paths) == N):
            print(f"[ERROR] Selected file counts do not match: Color ({len(color_paths)}), Depth ({len(depth_paths)}), PCD ({len(pcd_paths)}), Mask PCD ({len(mask_pcd_paths)}).")
            sys.exit(1)
        
        if N < 2:
            print("[ERROR] At least 2 frames (Source and Target) are required for registration.")
            sys.exit(1)

        # Load Intrinsic Matrix
        with open(intrinsic_path, 'r') as f:
            intrinsic_data = json.load(f)
        
        fx, fy, cx, cy = intrinsic_data['fx'], intrinsic_data['fy'], intrinsic_data['cx'], intrinsic_data['cy']
        width, height = intrinsic_data['width'], intrinsic_data['height']
        
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
        )
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Data loading or file selection failed: {e}")
        sys.exit(1)

    # 2. Odometry and Registration Loop
    print(f"\n--- 2. Starting Registration for {N} frames ({N-1} registrations) ---")
    
    # Initialize the cumulative transformation matrix (relative to Frame 0)
    cumulative_transform = np.identity(4)
    
    # Load the first point cloud (Frame 0)
    combined_pcd = o3d.io.read_point_cloud(pcd_paths[0])
    combined_mask_pcd = o3d.io.read_point_cloud(mask_pcd_paths[0])
    # Use the color map to assign a color to the first frame
    color_map = plt.colormaps['hsv']
    colors = color_map(np.linspace(0, 1, N)) 
    #combined_pcd.paint_uniform_color(colors[0][:3])
    
    # Parameters for RGBD loading and Odometry
    depth_scale = 1000.0 
    depth_trunc = 7.0 
    odometry_option = o3d.pipelines.odometry.OdometryOption(depth_min=0.1, depth_max=1.0)
    # Coarse-to-fine ICP parameters (meters)
    voxel_scales = [0.005, 0.003]
    max_corr_multiplier = 4.0
    

    for i in range(N - 1):
        frame_idx_i = i         # Reference Frame (Target for ICP)
        frame_idx_i_plus_1 = i + 1 # Moving Frame (Source for ICP)
        
        print(f"\n=======================================================")
        print(f"Processing Pair: Frame {frame_idx_i} -> Frame {frame_idx_i_plus_1}")
        
        # --- 2a. Load RGBD Data for Odometry ---
        # Note: Open3D expects Source (i) to Target (i+1) for Odometry
        source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(color_paths[frame_idx_i]), 
            o3d.io.read_image(depth_paths[frame_idx_i]), 
            depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(color_paths[frame_idx_i_plus_1]), 
            o3d.io.read_image(depth_paths[frame_idx_i_plus_1]), 
            depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        
        # --- 2b. Compute RGBD Odometry (Initial Transformation T_{i+1 <- i}) ---
        # Odometry estimates T_Source -> Target (Frame i -> Frame i+1)
        odo_init = np.identity(4)
        [success, odo_transform_i_to_i_plus_1, _] = o3d.pipelines.odometry.compute_rgbd_odometry(
             source_rgbd, target_rgbd, pinhole_camera_intrinsic, odo_init,
             o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), odometry_option
           )

        if not success:
            print(f"[WARNING] Odometry failed between Frame {frame_idx_i} and Frame {frame_idx_i_plus_1}. Skipping ICP and registration for this pair.")
            continue
            
        print(" -> Odometry Success! Used as initial guess for ICP.")
        
        # --- 2c. Prepare Data for ICP ---
        # We need the Point Clouds for ICP (PCD files are typically denser than those from RGBD creation)
        pcd_target = o3d.io.read_point_cloud(pcd_paths[frame_idx_i]) # Reference (Frame i)
        pcd_source = o3d.io.read_point_cloud(pcd_paths[frame_idx_i_plus_1]) # Moving (Frame i+1)
        
        # Compute the initial guess for T_{i+1 -> i} (aligns i+1 to i)
        # Odometry returns T_{i -> i+1}; invert for ICP which expects source (i+1) to target (i).
        odo_guess_i_plus_1_to_i = np.linalg.inv(odo_transform_i_to_i_plus_1)

        # Downsample for faster and more robust ICP convergence
        # --- 2d. Run Multi-Scale ICP Refinement ---
        print(" -> Running multi-scale ICP (coarse-to-fine)...")
        icp_transform_i_plus_1_to_i, reg_result = multi_scale_icp(
            source=pcd_source,
            target=pcd_target,
            initial_transform=odo_guess_i_plus_1_to_i,
            voxel_scales=voxel_scales,
            max_corr_mult=max_corr_multiplier
        )

        # --- 2e. Report ICP Metrics (Removed is_success check) ---
        print(f" -> ICP Result:")
        print(f"    Fitness (Overlap): {reg_result.fitness:.6f}")
        print(f"    Inlier RMSE (Error): {reg_result.inlier_rmse:.6f}")
        
        # --- 2f. Accumulate Transformation ---
        # Keep cumulative as T_{frame -> 0}; chain the ICP result directly (T_{i+1 -> i})
        cumulative_transform = cumulative_transform @ icp_transform_i_plus_1_to_i
        # Also keep the forward relative transform for logging (i -> i+1)
        icp_transform_i_to_i_plus_1 = np.linalg.inv(icp_transform_i_plus_1_to_i)

        # Log relative and global yaw to verify rotation is updating
        rot_rel = icp_transform_i_to_i_plus_1[:3, :3]
        yaw_rel = np.degrees(np.arctan2(rot_rel[1, 0], rot_rel[0, 0]))
        rot_global = cumulative_transform[:3, :3]
        yaw_global = np.degrees(np.arctan2(rot_global[1, 0], rot_global[0, 0]))
        print(f"   Relative Yaw (deg) i->i+1: {yaw_rel:.2f}")
        print(f"   Cumulative Yaw (deg) w.r.t Frame 0: {yaw_global:.2f}")
        print("   Cumulative Transform (i+1 -> 0):\n", cumulative_transform)

        # --- 2g. Transform and Combine Point Cloud ---
        current_pcd = o3d.io.read_point_cloud(pcd_paths[frame_idx_i_plus_1]) # Reload high-res PCD
        current_mask_pcd = o3d.io.read_point_cloud(mask_pcd_paths[frame_idx_i_plus_1])
        
        # Apply the full cumulative transform (Frame -> 0) to align the current frame 
        current_pcd.transform(cumulative_transform)
        current_mask_pcd.transform(cumulative_transform)
        
        # Apply a unique color for better visualization
        #current_pcd.paint_uniform_color(colors[frame_idx_i_plus_1][:3])
        #current_mask_pcd.paint_uniform_color(colors[frame_idx_i_plus_1][:3])
        
        # Add to the combined point cloud
        combined_pcd += current_pcd 
        combined_mask_pcd += current_mask_pcd 
        
    # --- 3. Final Visualization ---
    print("\n--- 3. Final Result Generation and Visualization ---")
    
    # Downsample the final combined point cloud for faster rendering/management
    combined_pcd_downsampled = combined_pcd.voxel_down_sample(voxel_size=0.01)
    combined_mask_pcd_downsampled = combined_mask_pcd.voxel_down_sample(voxel_size=0.01)
    
    print(f"Combined Point Cloud: {len(combined_pcd.points)} points total.")
    print(f"Downsampled Point Cloud: {len(combined_pcd_downsampled.points)} points.")
    print(f"Combined Mask Point Cloud: {len(combined_mask_pcd.points)} points total.")
    print(f"Downsampled Mask Point Cloud: {len(combined_mask_pcd_downsampled.points)} points.")

    # Save the final combined point clouds (full and mask)
    timestamp = int(time.time())
    output_dir = os.path.dirname(intrinsic_path)
    full_output_path = os.path.join(output_dir, f"combined_point_cloud_{timestamp}.ply")
    mask_output_path = os.path.join(output_dir, f"combined_mask_point_cloud_{timestamp}.ply")
    o3d.io.write_point_cloud(full_output_path, combined_pcd)
    o3d.io.write_point_cloud(mask_output_path, combined_mask_pcd)
    print(f"[SAVE] Combined point cloud saved to: {full_output_path}")
    print(f"[SAVE] Combined mask point cloud saved to: {mask_output_path}")
    
    # Show the final registered result
    visualize_geometry(
        [combined_pcd_downsampled, combined_mask_pcd_downsampled], 
        window_name=f"Final Registered Point Clouds ({N} Frames) - Odometry + ICP"
    )
    
    print("\nProcess finished. All frames successfully registered and combined.")

if __name__ == "__main__":
    try:
        run_odometry_registration()
    except Exception as e:
        # Handle exceptions gracefully
        print(f"\n[EXECUTION ERROR] An unexpected error occurred: {e}")
