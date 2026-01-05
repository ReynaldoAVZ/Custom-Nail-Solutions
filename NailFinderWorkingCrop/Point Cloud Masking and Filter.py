import numpy as np
import open3d as o3d
import cv2
import os
import tkinter as tk
from tkinter import filedialog
import sys

# --- 1. CAMERA INTRINSICS CONFIGURATION (MANDATORY TO REPLACE PLACEHOLDERS) ---
# NOTE: Replace the X's with the specific values retrieved from your RealSense logs
# (intr.fx, intr.fy, intr.ppx, intr.ppy).
FX = 385.47052001953125      # Focal length x
FY = 385.47052001953125      # Focal length y
CX = 326.9088134765625      # Principal point x
CY = 237.45928955078125      # Principal point y
IMAGE_WIDTH = 640  # Must match the resolution used during capture/masking
IMAGE_HEIGHT = 480 # Must match the resolution used during capture/masking

# Intrinsic Matrix K (3x3 array)
K = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# -----------------------------------------------------------------------------

def get_files_dialog(title, file_type_tuple):
    """Opens a file dialog to select multiple files."""
    root = tk.Tk()
    root.withdraw() # Hide the main window
    file_paths = filedialog.askopenfilenames(
        title=title,
        filetypes=[file_type_tuple]
    )
    # Destroy the root context to prevent side effects in some environments
    root.destroy()
    return list(file_paths)


def load_mask_and_pcd(pcd_path, mask_path):
    """Loads and preprocesses the mask image and point cloud."""
    
    # Load the point cloud
    try:
        pcd_raw = o3d.io.read_point_cloud(pcd_path)
        if not pcd_raw.has_points():
            print(f"File {os.path.basename(pcd_path)} loaded but contains no points.")
            return None, None
    except Exception as e:
        print(f"ERROR: Failed to load point cloud {os.path.basename(pcd_path)}: {e}")
        return None, None
        
    # Load the mask image
    try:
        # Load in grayscale (0)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(f"Mask image not found at {mask_path}")
            
        # Check and resize mask dimensions (essential for correct projection)
        if mask_img.shape[0] != IMAGE_HEIGHT or mask_img.shape[1] != IMAGE_WIDTH:
             print(f"  WARNING: Mask size {mask_img.shape} resized to {IMAGE_HEIGHT}x{IMAGE_WIDTH}.")
             mask_img = cv2.resize(mask_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

        # Ensure mask is binary (0 or 255)
        mask_img[mask_img > 0] = 255
        
    except Exception as e:
        print(f"ERROR: Failed to load or process mask image {os.path.basename(mask_path)}: {e}")
        return None, None

    return pcd_raw, mask_img


def project_and_filter(pcd, mask_img, K_matrix):
    """
    Projects 3D points onto the 2D image plane and filters points based on the mask.
    Returns the filtered point cloud in Open3D format.
    """
    points_3d = np.asarray(pcd.points)
    
    # 1. Prepare points (N, 3) -> (3, N)
    points_camera_frame = points_3d.T 

    # 2. Project points to homogeneous 2D coordinates (3, N)
    projected_homogeneous = K_matrix @ points_camera_frame

    # 3. Normalize to get pixel coordinates (u, v)
    depths = projected_homogeneous[2, :]
    
    # Filter points with invalid depth
    valid_depth_mask = depths > 0.001 
    points_3d_valid = points_3d[valid_depth_mask]
    depths_valid = depths[valid_depth_mask]
    projected_homogeneous_valid = projected_homogeneous[:, valid_depth_mask]
    
    u = projected_homogeneous_valid[0, :] / depths_valid
    v = projected_homogeneous_valid[1, :] / depths_valid

    # 4. Check boundaries and mask values
    H, W = mask_img.shape
    
    # Check if projected points are within the image boundaries
    valid_u = np.logical_and(u >= 0, u < W)
    valid_v = np.logical_and(v >= 0, v < H)
    is_in_bounds = np.logical_and(valid_u, valid_v)

    # Convert u, v to integer indices (v is row index, u is col index)
    u_int = u.astype(int)
    v_int = v.astype(int)

    # Filter bounded points and their corresponding 3D data
    points_3d_bounded = points_3d_valid[is_in_bounds]
    u_bounded = u_int[is_in_bounds]
    v_bounded = v_int[is_in_bounds]

    # Get the mask value at the projected pixel location
    mask_values = mask_img[v_bounded, u_bounded]

    # Filter: Keep points where the mask is white (non-zero)
    is_on_mask = mask_values > 0

    # 5. Extract the final filtered point cloud
    filtered_points_array = points_3d_bounded[is_on_mask]
    
    # Convert back to Open3D format
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points_array)

    return filtered_pcd


if __name__ == "__main__":
    
    print("--- Starting Batch Point Cloud Masking ---")
    print("1. Select all PLY Point Cloud files.")
    pc_files = get_files_dialog("Select Point Cloud Files (.ply)", ('Point Clouds', '*.ply'))
    
    if not pc_files:
        print("No point cloud files selected. Exiting.")
        sys.exit()
        
    print("2. Select all PNG Mask files (Must be in the same order as the PLY files).")
    mask_files = get_files_dialog("Select Mask Files (.png)", ('Mask Images', '*.png'))
    
    if len(pc_files) != len(mask_files):
        print(f"\nERROR: File count mismatch!")
        print(f"Selected {len(pc_files)} Point Clouds and {len(mask_files)} Masks.")
        print("Please restart and select an equal number of files in corresponding order.")
        sys.exit()

    print(f"\nSuccessfully paired {len(pc_files)} sets of files. Starting processing...")
    
    # Process each pair
    for i, (pc_path, mask_path) in enumerate(zip(pc_files, mask_files)):
        pc_filename = os.path.basename(pc_path)
        mask_filename = os.path.basename(mask_path)
        print(f"\n--- Processing Pair {i+1}/{len(pc_files)}: PC={pc_filename}, Mask={mask_filename} ---")

        # 1. Load Data
        pcd_raw, mask_img = load_mask_and_pcd(pc_path, mask_path)
        
        if pcd_raw is None:
            print(f"Skipping {pc_filename} due to loading error.")
            continue
        
        # 2. Filter Point Cloud
        filtered_pcd = project_and_filter(pcd_raw, mask_img, K)
        
        filtered_count = len(filtered_pcd.points)
        print(f"Original Points: {len(pcd_raw.points)}")
        print(f"Filtered Nail Points: {filtered_count}")

        if filtered_count < 100:
            print("  WARNING: Very few points remaining. Check your K matrix or mask.")
            
        # 3. Save the filtered result
        if filtered_count > 0:
            # Create the new filename: /path/to/my_file.ply -> /path/to/my_file_masked.ply
            base_name, ext = os.path.splitext(pc_path)
            output_path = f"{base_name}_masked.ply"
            
            o3d.io.write_point_cloud(output_path, filtered_pcd)
            print(f"Saved successful masked segment to: {os.path.basename(output_path)}")
        else:
            print(f"Skipping save for {pc_filename} as no points were filtered.")
            
    print("\n--- Batch Processing Complete ---")

    # NOTE: You can uncomment the following lines if you want to visually check
    # the last processed filtered point cloud for debugging purposes.
    # if 'filtered_pcd' in locals() and filtered_count > 0:
    #     print("Visualizing the last filtered point cloud...")
    #     coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1) 
    #     filtered_pcd.paint_uniform_color([0, 0.8, 0])
    #     o3d.visualization.draw_geometries([filtered_pcd, coord_frame], 
    #                                      window_name="Last Filtered Nail Points")