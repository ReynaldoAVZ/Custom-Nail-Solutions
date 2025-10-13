import numpy as np
import open3d as o3d
from tkinter import Tk, filedialog
import os
from matplotlib import pyplot as plt
import copy
import open3d.visualization as vis
import time

# Hide the main tkinter window
root = Tk()
root.withdraw()

# Ask user to select the color .npy file
print("Select the color image (.npy)")
color_path = filedialog.askopenfilename(title="Select Color Image (.npy)", filetypes=[("NumPy files", "*.npy")])

# Ask user to select the depth .npy file
print("Select the depth image (.npy)")
depth_path = filedialog.askopenfilename(title="Select Depth Image (.npy)", filetypes=[("NumPy files", "*.npy")])

# Load color and depth from selected .npy files
color_array = np.load(color_path)  # Expect shape (H, W, 3), dtype=uint8
depth_array = np.load(depth_path)  # Expect shape (H, W), dtype=uint16 or float

# Convert numpy arrays to Open3D image objects
color_o3d = o3d.geometry.Image(color_array.astype(np.uint8))
depth_o3d = o3d.geometry.Image(depth_array.astype(np.uint16))  # or float if needed

# Create RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d,
    depth_o3d
)

print(rgbd_image)

# --- Depth Image Contrast Enhancement ---
# 1. Clamping: Limit the depth range to a relevant interval (e.g., based on the object of interest).
depth_data = np.asarray(rgbd_image.depth)
min_depth = np.percentile(depth_data[depth_data > 0], 5)  # Example: 5th percentile, exclude 0
max_depth = np.percentile(depth_data[depth_data > 0], 95)  # Example: 95th percentile
clamped_depth_data = np.clip(depth_data, min_depth, max_depth)

# 2. Normalization: Scale the depth values to the range [0, 1] for better visualization.
normalized_depth_data = (clamped_depth_data - clamped_depth_data.min()) / (
    clamped_depth_data.max() - clamped_depth_data.min()
)

# Plotting
plt.figure(figsize=(12, 6))

# Show color image
plt.subplot(1, 2, 1)
plt.title('Finger RGB image')
plt.imshow(np.asarray(rgbd_image.color))
plt.axis('off')

# Show enhanced depth image with legend
plt.subplot(1, 2, 2)
plt.title('Enhanced Finger depth image')
depth_img = plt.imshow(normalized_depth_data, cmap='plasma')  # Use normalized data
plt.axis('off')
cbar = plt.colorbar(depth_img, fraction=0.046, pad=0.04)
cbar.set_label('Depth (normalized)')  # Update label

plt.tight_layout()
plt.show()

# Create point cloud from RGBD image
# Ask the user to select the intrinsics file
print("Select the intrinsics.npy file")
intrinsics_path = filedialog.askopenfilename(
    title="Select intrinsics.npy", filetypes=[("NumPy files", "*.npy")]
)

# Load the intrinsics dictionary
intrinsics_dict = np.load(intrinsics_path, allow_pickle=True).item()

# Create Open3D PinholeCameraIntrinsic from the dict
custom_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    intrinsics_dict['width'],
    intrinsics_dict['height'],
    intrinsics_dict['fx'],
    intrinsics_dict['fy'],
    intrinsics_dict['ppx'],
    intrinsics_dict['ppy']
)

# Create point cloud from RGBD image using your intrinsics
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    custom_intrinsics
)

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])

# use the nearest neighbor stuff
# Step 1: Find the center of the point cloud (mean of all points)
points = np.asarray(pcd.points)
center_point = points.mean(axis=0)

# Step 2: Use KDTree to find nearest neighbors to the center
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
k = 50  # Number of nearest neighbors to highlight
_, idx, _ = pcd_tree.search_knn_vector_3d(center_point, k)

# Step 3: Paint the point cloud - make a copy so original isn't altered
colored_pcd = copy.deepcopy(pcd)
colors = np.asarray(colored_pcd.colors)

# Paint all points light gray first (optional, for contrast)
colors[:] = [0.7, 0.7, 0.7]

# Paint nearest neighbors red
for i in idx:
    colors[i] = [1.0, 0.0, 0.0]

# Visualize
o3d.visualization.draw_geometries([colored_pcd])

import time

print("Applying statistical outlier removal with multiple combinations...\n")

# Define 5 sample values for each parameter
nb_neighbors_list = [100, 10, 20, 30, 40, 50]
std_ratio_list = [.1, .25, 0.5, 1.0, 1.5, 2.0, 2.5]

# Downsample first to improve visualization performance
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.005)

# Initialize the visualizer once
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Statistical Outlier Removal", width=800, height=600)

geom_added = False

for nb in nb_neighbors_list:
    for std in std_ratio_list:
        print(f"Filtering with nb_neighbors={nb}, std_ratio={std}")
        cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)

        inlier_cloud = voxel_down_pcd.select_by_index(ind)
        outlier_cloud = voxel_down_pcd.select_by_index(ind, invert=True)

        inlier_cloud.paint_uniform_color([0.7, 0.7, 0.7])
        outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])

        filtered_combined = inlier_cloud + outlier_cloud

        if not geom_added:
            vis.add_geometry(filtered_combined)
            geom_added = True
        else:
            vis.clear_geometries()
            vis.add_geometry(filtered_combined)

        vis.poll_events()
        vis.update_renderer()
        vis.get_render_option().point_size = 2.0

        print(f"Showing: nb_neighbors={nb}, std_ratio={std}")
        time.sleep(2)  # Wait before moving to next combo

# Close visualizer after all combinations
vis.destroy_window()
