import open3d as o3d
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
import statistics
import os
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from matplotlib.cm import ScalarMappable  # Import for colorbar


def analyze_point_cloud(file_path, is_npy=False, intrinsics=None):
    """
    Analyzes a single point cloud file for dimensional accuracy, processing time,
    resolution, repeatability, and surface smoothness.  Handles both .ply/.stl/.obj/.pcd and .npy files.

    Args:
        file_path (str): Path to the point cloud file.
        is_npy (bool, optional):  True if the file is a pair of .npy files (RGB and depth). Defaults to False.
        intrinsics (dict, optional): Camera intrinsics if is_npy is True.  Required if is_npy is True.
    Returns:
        dict: A dictionary containing the analysis results.  Returns None if there is an error.
    """
    try:
        if is_npy:
            if intrinsics is None:
                raise ValueError("Intrinsics must be provided when analyzing .npy files.")
            # Load point cloud from .npy files
            rgb_file = file_path[0]  # Assume file_path is a tuple of (rgb_file, depth_file)
            depth_file = file_path[1]
            pcd = generate_point_cloud_from_npy(rgb_file, depth_file, intrinsics)
        else:
            # Load point cloud
            pcd = o3d.io.read_point_cloud(file_path)

        if pcd.is_empty():
            print(f"Error: Point cloud file is empty: {file_path}")
            return None

        points = np.asarray(pcd.points)

        # 1. Dimensional Accuracy (BEFORE filtering)
        min_bounds_before = points.min(axis=0)
        max_bounds_before = points.max(axis=0)
        dimensions_m_before = max_bounds_before - min_bounds_before
        dimensions_mm_before = dimensions_m_before * 1000
        dimensions_result_before = dimensions_mm_before.tolist()  # Convert to list

        # 2. Processing Time
        start_time = time.time()
        # Simulated processing steps (replace with actual processing if needed)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)  # Store the downsampled PCD
        processing_time = time.time() - start_time

        points_downsampled = np.asarray(pcd_downsampled.points)  # get points after downsampling

        # 1. Dimensional Accuracy (AFTER filtering)
        min_bounds = points_downsampled.min(axis=0)
        max_bounds = points_downsampled.max(axis=0)
        dimensions_m = max_bounds - min_bounds
        dimensions_mm = dimensions_m * 1000
        dimensions_result = dimensions_mm.tolist()  # Convert to list for easier handling

        # 3. Resolution (points/mm²)
        try:
            pcd.estimate_normals()  # Estimate normals for original point cloud
            mesh_before, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            area_m2_before = mesh_before.get_surface_area()
            num_points_before = len(points)
            resolution_before = num_points_before / (area_m2_before * 1e6)
        except Exception as e:
            print(f"Resolution calculation failed for {file_path} (before filtering): {e}")
            resolution_before = None

        try:
            pcd_downsampled.estimate_normals()
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_downsampled, depth=8)
            area_m2 = mesh.get_surface_area()
            num_points = len(points_downsampled)
            resolution = num_points / (area_m2 * 1e6)
        except Exception as e:
            print(f"Resolution calculation failed for {file_path} (after filtering): {e}")
            resolution = None

        # 4. Repeatability (Mock comparison to itself)
        distances = pcd_downsampled.compute_point_cloud_distance(pcd_downsampled)
        if len(distances) > 0:
            repeatability = np.mean(distances) * 1000  # in mm
        else:
            repeatability = None

        # 5. Surface Smoothness (RMS Roughness)
        # Calculate before filtering
        pcd_smoothed_before = pcd.voxel_down_sample(voxel_size=0.001)
        points_smoothed_before = np.asarray(pcd_smoothed_before.points)
        if points_smoothed_before.size > 0:
            mean_point_before = points_smoothed_before.mean(axis=0)
            rms_roughness_before = np.sqrt(((points_smoothed_before - mean_point_before) ** 2).mean()) * 1000
        else:
            rms_roughness_before = None

        # Calculate after filtering
        pcd_smoothed = pcd_downsampled.voxel_down_sample(voxel_size=0.001)  # Use the downsampled point cloud
        points_smoothed = np.asarray(pcd_smoothed.points)
        if points_smoothed.size > 0:
            mean_point = points_smoothed.mean(axis=0)
            rms_roughness = np.sqrt(((points_smoothed - mean_point) ** 2).mean()) * 1000
        else:
            rms_roughness = None

        return {
            "file_name": os.path.basename(file_path[0] if is_npy else file_path),  # Use RGB name for npy
            "dimensions_mm_before": dimensions_result_before,  # Store the dimensions before
            "dimensions_mm": dimensions_result,
            "processing_time": processing_time,
            "resolution_before": resolution_before, # Store resolution before
            "resolution": resolution,
            "repeatability": repeatability,
            "rms_roughness_before": rms_roughness_before, #store rms before
            "rms_roughness": rms_roughness,
            "pcd": pcd,  # returning the original point cloud
            "pcd_downsampled": pcd_downsampled  # returning the downsampled point cloud
        }
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None


def select_files():
    """
    Opens a file dialog to select multiple point cloud files (.ply, .stl, .obj, .pcd) or .npy pairs.

    Returns:
        list: A list of file paths or a list of tuples of (rgb_file, depth_file) if .npy files are selected.
        bool: True if .npy files were selected, False otherwise.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(filetypes=[("Point Cloud Files", "*.ply;*.stl;*.obj;*.pcd;*.npy")])
    root.destroy()
    npy_files = [f for f in file_paths if f.lower().endswith(".npy")]
    other_files = [f for f in file_paths if not f.lower().endswith(".npy")]

    if npy_files:
        rgb_files, depth_files = select_npy_files()
        return list(zip(rgb_files, depth_files)), True
    else:
        return other_files, False


def select_npy_files():
    """
    Selects pairs of RGB and depth .npy files.  This function assumes that the files
    are named in such a way that it can distinguish between RGB and depth images
    based on the filename.
    """
    print("Select all the RGB and depth .npy files (hold CTRL or SHIFT to select multiple)...")
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select RGB and Depth .npy files",
        filetypes=[("NumPy files", "*.npy")]
    )
    root.destroy()

    rgb_files = sorted([f for f in file_paths if 'color' in os.path.basename(f).lower() or 'rgb' in os.path.basename(f).lower()])
    depth_files = sorted([f for f in file_paths if 'depth' in os.path.basename(f).lower()])

    if len(rgb_files) != len(depth_files):
        raise ValueError("Number of RGB and depth files must match!")

    print(f"Selected {len(rgb_files)} frame pairs.")
    return rgb_files, depth_files


def load_intrinsics_from_file(intrinsics_file):
    """Loads camera intrinsics from a .npy file or returns default values."""
    if os.path.exists(intrinsics_file):
        return np.load(intrinsics_file, allow_pickle=True).item()
    else:
        print("Intrinsics file not found. Using default intrinsics.")
        return {
            'width': 640,
            'height': 480,
            'fx': 616.36529541,
            'fy': 616.20294189,
            'ppx': 310.25881958,
            'ppy': 236.59980774,
            'depth_scale': 0.001
        }


def generate_point_cloud_from_npy(rgb_file, depth_file, intrinsics):
    """Generates a point cloud from RGB and depth .npy files using given intrinsics."""
    color = np.load(rgb_file)
    depth = np.load(depth_file)

    if color.shape[:2] != depth.shape:
        raise ValueError(f"Resolution mismatch for {rgb_file} and {depth_file}")

    height, width = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    ppx, ppy = intrinsics['ppx'], intrinsics['ppy']
    scale = intrinsics['depth_scale']

    points = []
    colors = []

    for y in range(height):
        for x in range(width):
            z = depth[y, x] * scale
            if z == 0:
                continue
            x3d = (x - ppx) * z / fx
            y3d = (y - ppy) * z / fy
            points.append([x3d, y3d, z])
            colors.append(color[y, x] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd


def analyze_and_display(file_paths, is_npy, intrinsics=None):
    """
    Analyzes the selected point cloud files and displays the results,
    including statistical analysis and plots.  Handles both .ply/.stl/.obj/.pcd and .npy files.

    Args:
        file_paths (list): A list of paths to the point cloud files to analyze.
                            If is_npy is True, this is a list of tuples: (rgb_file, depth_file).
        is_npy (bool): True if the files are .npy files, False otherwise.
        intrinsics (dict, optional): Camera intrinsics if is_npy is True.
    """
    results = []
    if is_npy and intrinsics is None:
        print("Error: Intrinsic parameters are required for .npy files.")
        return

    for file_path in file_paths:
        result = analyze_point_cloud(file_path, is_npy, intrinsics)
        if result:  # Only add valid results
            results.append(result)

    if not results:
        print("No valid point cloud files selected or errors occurred during analysis.")
        return

    # Print detailed results for each file
    print("\nDetailed Results for Each File:")
    for result in results:
        print(f"\nFile: {result['file_name']}")
        print(f"  Dimensions Before Filtering (mm): {result['dimensions_mm_before']}")
        print(f"  Dimensions After Filtering (mm): {result['dimensions_mm']}")
        print(f"  Processing Time (s): {result['processing_time']:.4f}")
        if result['resolution_before'] is not None:
            print(f"  Resolution Before Filtering (points/mm²): {result['resolution_before']:.2f}")
        else:
            print("  Resolution Before Filtering: N/A")
        if result['resolution'] is not None:
            print(f"  Resolution After Filtering (points/mm²): {result['resolution']:.2f}")
        else:
            print("  Resolution After Filtering: N/A")
        if result['repeatability'] is not None:
            print(f"  Repeatability Deviation (mm): {result['repeatability']:.4f}")
        else:
            print("  Repeatability Deviation: N/A")
        if result['rms_roughness_before'] is not None:
            print(f"  Surface RMS Roughness Before Filtering (mm): {result['rms_roughness_before']:.4f}")
        else:
            print("  Surface RMS Roughness Before Filtering: N/A")
        if result['rms_roughness'] is not None:
            print(f"  Surface RMS Roughness After Filtering (mm): {result['rms_roughness']:.4f}")
        else:
            print("  Surface RMS Roughness After Filtering: N/A")

    # Statistical Analysis
    print("\n--- Statistical Analysis ---")

    # Before Filtering
    print("\nBefore Filtering:")
    dimensions_x_before = [result['dimensions_mm_before'][0] for result in results if result['dimensions_mm_before'] is not None]
    dimensions_y_before = [result['dimensions_mm_before'][1] for result in results if result['dimensions_mm_before'] is not None]
    dimensions_z_before = [result['dimensions_mm_before'][2] for result in results if result['dimensions_mm_before'] is not None]
    resolution_values_before = [result['resolution_before'] for result in results if result['resolution_before'] is not None]
    rms_roughness_values_before = [result['rms_roughness_before'] for result in results if result['rms_roughness_before'] is not None]

    if dimensions_x_before:
        print(f"Dimensions X: Mean = {statistics.mean(dimensions_x_before):.4f} mm, StDev = {statistics.stdev(dimensions_x_before):.4f} mm")
    else:
        print("Dimensions X: Not enough data to calculate statistics")
    if dimensions_y_before:
        print(f"Dimensions Y: Mean = {statistics.mean(dimensions_y_before):.4f} mm, StDev = {statistics.stdev(dimensions_y_before):.4f} mm")
    else:
        print("Dimensions Y: Not enough data to calculate statistics")
    if dimensions_z_before:
        print(f"Dimensions Z: Mean = {statistics.mean(dimensions_z_before):.4f} mm, StDev = {statistics.stdev(dimensions_z_before):.4f} mm")
    else:
        print("Dimensions Z: Not enough data to calculate statistics")
    if resolution_values_before:
        print(
            f"Resolution: Mean = {statistics.mean(resolution_values_before):.4f} points/mm², StDev = {statistics.stdev(resolution_values_before):.4f} points/mm²")
    else:
        print("Resolution: Not enough data to calculate statistics")
    if rms_roughness_values_before:
        print(f"Surface Roughness: Mean = {statistics.mean(rms_roughness_values_before):.4f} mm, StDev = {statistics.stdev(rms_roughness_values_before):.4f} mm")
    else:
        print("Surface Roughness: Not enough data to calculate statistics.")

    # After Filtering
    print("\nAfter Filtering:")
    repeatability_values = [result['repeatability'] for result in results if result['repeatability'] is not None]
    rms_roughness_values = [result['rms_roughness'] for result in results if result['rms_roughness'] is not None]
    dimensions_x = [result['dimensions_mm'][0] for result in results if result['dimensions_mm'] is not None]
    dimensions_y = [result['dimensions_mm'][1] for result in results if result['dimensions_mm'] is not None]
    dimensions_z = [result['dimensions_mm'][2] for result in results if result['dimensions_mm'] is not None]
    resolution_values = [result['resolution'] for result in results if result['resolution'] is not None]

    if repeatability_values:
        print(f"Repeatability: Mean = {statistics.mean(repeatability_values):.4f} mm, StDev = {statistics.stdev(repeatability_values):.4f} mm")
    else:
        print("Repeatability: Not enough data to calculate statistics.")

    if rms_roughness_values:
        print(f"Surface Roughness: Mean = {statistics.mean(rms_roughness_values):.4f} mm, StDev = {statistics.stdev(rms_roughness_values):.4f} mm")
    else:
        print("Surface Roughness: Not enough data to calculate statistics.")
    if dimensions_x:
        print(f"Dimensions X: Mean = {statistics.mean(dimensions_x):.4f} mm, StDev = {statistics.stdev(dimensions_x):.4f} mm")
    else:
        print("Dimensions X: Not enough data to calculate statistics")
    if dimensions_y:
        print(f"Dimensions Y: Mean = {statistics.mean(dimensions_y):.4f} mm, StDev = {statistics.stdev(dimensions_y):.4f} mm")
    else:
        print("Dimensions Y: Not enough data to calculate statistics")
    if dimensions_z:
        print(f"Dimensions Z: Mean = {statistics.mean(dimensions_z):.4f} mm, StDev = {statistics.stdev(dimensions_z):.4f} mm")
    else:
        print("Dimensions Z: Not enough data to calculate statistics")
    if resolution_values:
        print(f"Resolution: Mean = {statistics.mean(resolution_values):.4f} points/mm², StDev = {statistics.stdev(resolution_values):.4f} points/mm²")
    else:
        print("Resolution: Not enough data to calculate statistics")

    # Visualize the results
    visualize_results(results)



def visualize_results(results):
    """
    Visualizes the analysis results using matplotlib.  Handles point cloud differences
    more effectively by showing pairwise comparisons.

    Args:
        results (list): A list of dictionaries, where each dictionary contains the analysis
                        results for a single file.
    """
    num_files = len(results)

    # 1. Dimensions Plot (X, Y, Z on separate plots)
    plt.figure(figsize=(18, 12))  # Increased figure size for better subplot spacing

    plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
    plt.plot(range(num_files), [result['dimensions_mm_before'][0] for result in results], marker='o',
             label='X Before')
    plt.plot(range(num_files), [result['dimensions_mm'][0] for result in results], marker='x',
             label='X After')
    plt.xlabel('File Index')
    plt.ylabel('X Dimension (mm)')
    plt.title('X Dimensions')
    plt.legend()
    plt.grid(True)  # Add gridlines

    plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
    plt.plot(range(num_files), [result['dimensions_mm_before'][1] for result in results], marker='o',
             label='Y Before')
    plt.plot(range(num_files), [result['dimensions_mm'][1] for result in results], marker='x',
             label='Y After')
    plt.xlabel('File Index')
    plt.ylabel('Y Dimension (mm)')
    plt.title('Y Dimensions')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
    plt.plot(range(num_files), [result['dimensions_mm_before'][2] for result in results], marker='o',
             label='Z Before')
    plt.plot(range(num_files), [result['dimensions_mm'][2] for result in results], marker='x',
             label='Z After')
    plt.xlabel('File Index')
    plt.ylabel('Z Dimension (mm)')
    plt.title('Z Dimensions')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()

    # 2. Resolution Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_files), [result['resolution_before'] for result in results], marker='o', label='Resolution Before')
    plt.plot(range(num_files), [result['resolution'] for result in results], marker='x', label='Resolution After')
    plt.xlabel('File Index')
    plt.ylabel('Resolution (points/mm²)')
    plt.title('Resolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Surface Roughness Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_files), [result['rms_roughness_before'] for result in results], marker='o', label='RMS Before')
    plt.plot(range(num_files), [result['rms_roughness'] for result in results], marker='x', label='RMS After')
    plt.xlabel('File Index')
    plt.ylabel('RMS Roughness (mm)')
    plt.title('Surface Roughness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Point Cloud Differences (Pairwise Comparison)
    if num_files > 1:
        variances = [] # To store variance
        for i in range(num_files):
            for j in range(i + 1, num_files):  # Avoid redundant comparisons
                print(f"\nComparing Point Clouds: File {i+1} vs. File {j+1}")
                pcd1 = results[i]['pcd']
                pcd2 = results[j]['pcd_downsampled'] #compare to the downsampled version

                # Calculate distance and variance
                distance, variance = calculate_point_cloud_distance_and_variance(pcd1, pcd2) # Changed function
                variances.append(variance) # Append
                print(f"Distance: {distance:.4f} mm, Variance: {variance:.4f} mm²")

                # Visualize the comparison
                fig = plt.figure(figsize=(12, 6))
                ax1 = fig.add_subplot(121, projection='3d')
                ax2 = fig.add_subplot(122, projection='3d')

                ax1.set_title(f'File {i+1} (Original)')
                ax1.scatter(np.asarray(pcd1.points)[:, 0], np.asarray(pcd1.points)[:, 1], np.asarray(pcd1.points)[:, 2], c='blue', s=10)
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')

                ax2.set_title(f'File {j+1} (Downsampled)')
                ax2.scatter(np.asarray(pcd2.points)[:, 0], np.asarray(pcd2.points)[:, 1], np.asarray(pcd2.points)[:, 2], c='green', s=10)
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('Z')
                plt.tight_layout()
                plt.show()

                # Visualize Point Cloud differences
                visualize_point_cloud_diff(pcd1, pcd2)
        # Plot the variances
        plt.figure()
        plt.plot(range(len(variances)), variances, marker='o')
        plt.xlabel('Pairwise Comparison Index')
        plt.ylabel('Variance (mm²)')
        plt.title('Variance of Pairwise Point Cloud Distances')
        plt.show()

def calculate_point_cloud_distance_and_variance(pcd1, pcd2):
    """
    Calculates a simplified distance metric and variance between two point clouds.
    If the point clouds have different numbers of points, it returns a large number
    for distance and a large number for variance.
    Otherwise, it calculates the mean Euclidean distance and variance between
    corresponding points.

    Args:
        pcd1 (open3d.geometry.PointCloud): The first point cloud.
        pcd2 (open3d.geometry.PointCloud): The second point cloud.

    Returns:
        tuple (float, float): The mean Euclidean distance and the variance of the
                            distances, or a large number for both if the point
                            clouds have different sizes.
    """
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    if len(points1) != len(points2):
        return 1e10, 1e10  # Return large numbers for both distance and variance
    else:
        distances = np.linalg.norm(points1 - points2, axis=1)
        mean_distance = np.mean(distances)
        variance = np.var(distances)
        return mean_distance, variance

def visualize_point_cloud_diff(pcd1, pcd2):
    """
    Visualizes the difference between two point clouds by calculating the distance
    between corresponding points (if they have the same number of points) and
    coloring the points in the first point cloud based on this distance.  If the
    point clouds have different numbers of points, it visualizes them separately.

    Args:
        pcd1 (open3d.geometry.PointCloud): The first point cloud.
        pcd2 (open3d.geometry.PointCloud): The second point cloud.
    """
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Point Cloud Difference')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if len(points1) == len(points2):
        # Calculate Euclidean distances between corresponding points
        distances = np.linalg.norm(points1 - points2, axis=1)
        
        # Color points based on distance (e.g., using a colormap)
        cmap = plt.get_cmap('jet')  # You can choose a different colormap
        norm = plt.Normalize(distances.min(), distances.max())
        colors = cmap(norm(distances))
        
        # Plot the first point cloud with colors representing the distance
        scatter = ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c=colors, s=10)
        
        # Create a colorbar to show the distance-to-color mapping
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Important:  You need to set an empty array for the ScalarMappable
        fig.colorbar(sm, ax=ax, label='Distance')
    else:
        # If the point clouds have different numbers of points, plot them separately
        print("Point clouds have different number of points, plotting separately.")
        ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='blue', s=10, label='PointCloud1')
        ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='green', s=10, label='PointCloud2')
        ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    intrinsics_file = "C:/Users/reyna/source/repos/Custom-Nail-Solutions/wrappers/python/examples/Combined Process/intrinsics.npy"
    intrinsics = load_intrinsics_from_file(intrinsics_file)

    file_paths, is_npy = select_files()
    if file_paths:
        analyze_and_display(file_paths, is_npy, intrinsics)
    else:
        print("No files selected.")
