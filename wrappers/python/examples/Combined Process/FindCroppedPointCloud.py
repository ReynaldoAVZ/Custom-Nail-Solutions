import numpy as np
import open3d as o3d
from tkinter import Tk, filedialog
import os

def select_npy_files():
    print("Select all the RGB and depth .npy files (hold CTRL or SHIFT to select multiple)...")
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select RGB and Depth .npy files",
        filetypes=[("NumPy files", "*.npy")]
    )

    rgb_files = sorted([f for f in file_paths if 'color' in os.path.basename(f).lower()])
    depth_files = sorted([f for f in file_paths if 'depth' in os.path.basename(f).lower()])

    if len(rgb_files) != len(depth_files):
        raise ValueError("Number of RGB and depth files must match!")

    print(f"Selected {len(rgb_files)} frame pairs.")
    return rgb_files, depth_files

def load_intrinsics_from_file(intrinsics_file):
    if os.path.exists(intrinsics_file):
        return np.load(intrinsics_file, allow_pickle=True).item()
    else:
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

def align_point_clouds(pcds):
    print("\nAligning point clouds using ICP...")
    if len(pcds) < 2:
        print("Not enough point clouds to align. Returning single cloud.")
        return pcds[0]

    aligned_pcd = pcds[0]
    for i in range(1, len(pcds)):
        print(f" - Aligning frame {i}...")
        prev_center = np.mean(np.asarray(aligned_pcd.points), axis=0)
        curr_center = np.mean(np.asarray(pcds[i].points), axis=0)
        init_guess = np.eye(4)
        init_guess[:3, 3] = prev_center - curr_center

        reg = o3d.pipelines.registration.registration_icp(
            pcds[i], aligned_pcd, 0.02, init_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        pcds[i].transform(reg.transformation)
        aligned_pcd += pcds[i]

    print("ICP alignment complete.")
    return aligned_pcd

def filter_top_layer(pcd, voxel_size=0.001):
    print("Filtering to keep only top layer points...")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    xy_grid = np.round(points[:, :2] / voxel_size, decimals=3)
    xy_tuples = [tuple(xy) for xy in xy_grid]

    min_z_dict = {}
    min_z_color = {}

    for i, key in enumerate(xy_tuples):
        z = points[i, 2]
        if key not in min_z_dict or z < min_z_dict[key]:
            min_z_dict[key] = z
            min_z_color[key] = colors[i]

    filtered_points = []
    filtered_colors = []

    for key in min_z_dict:
        x, y = np.array(key) * voxel_size
        z = min_z_dict[key]
        filtered_points.append([x, y, z])
        filtered_colors.append(min_z_color[key])

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
    filtered_pcd.colors = o3d.utility.Vector3dVector(np.array(filtered_colors))

    print(f"Top layer filtering complete. Reduced from {len(points)} to {len(filtered_points)} points.")
    return filtered_pcd

def apply_filters(pcd, voxel_size=0.0005):
    print("\nApplying filters...")
    print(f"Initial point count: {len(pcd.points)}")
    
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.5)
    # pcd = filter_top_layer(pcd, voxel_size=0.001)
    print("All filtering complete.")
    return pcd

def estimate_normals(pcd, radius=0.01, max_nn=30):
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(100)
    return pcd

def filter_by_curvature(pcd, threshold=.5):
    print("Filtering by curvature...")
    normals = np.asarray(pcd.normals)
    curvature = np.linalg.norm(normals - np.mean(normals, axis=0), axis=1)
    mask = curvature < threshold
    filtered_pcd = pcd.select_by_index(np.where(mask)[0])
    print(f"Curvature filtering removed {len(pcd.points) - len(filtered_pcd.points)} points.")
    return filtered_pcd

def segment_planar_surface(pcd, distance_threshold=0.5, ransac_n=3, num_iterations=1000):
    print("Segmenting planar surface...")
    if len(pcd.points) < ransac_n:
        print(f"Not enough points ({len(pcd.points)}) for plane segmentation. Skipping this step.")
        return pcd

    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    print(f"Plane segmented. Kept {len(inliers)} points.")
    return inlier_cloud

def save_point_cloud(pcd, filename, file_format="ply"):
    output_file = f"{filename}.{file_format}"
    print(f"\nSaving point cloud to '{output_file}'...")
    o3d.io.write_point_cloud(output_file, pcd)
    print("Saved successfully.")

def save_mesh(mesh, filename):
    output_file = f"{filename}_mesh.stl"
    print(f"\nSaving mesh to '{output_file}'...")
    o3d.io.write_triangle_mesh(output_file, mesh)
    
    print("Mesh saved successfully.")

def main():
    intrinsics_file = "C:/Users/reyna/source/repos/Custom-Nail-Solutions/wrappers/python/examples/Combined Process/intrinsics.npy"
    intrinsics = load_intrinsics_from_file(intrinsics_file)

    rgb_files, depth_files = select_npy_files()

    pcds = []
    for rgb_file, depth_file in zip(rgb_files, depth_files):
        print(f"Processing {os.path.basename(rgb_file)} + {os.path.basename(depth_file)}...")
        pcd = generate_point_cloud_from_npy(rgb_file, depth_file, intrinsics)
        pcds.append(pcd)

    merged_pcd = align_point_clouds(pcds)
    filtered_pcd = apply_filters(merged_pcd)

    # Extra filtering steps
    filtered_pcd = estimate_normals(filtered_pcd)
    filtered_pcd = filter_by_curvature(filtered_pcd, threshold=1)
    #filtered_pcd = segment_planar_surface(filtered_pcd)

    base_filename = input("Enter base filename for point cloud (no extension): ").strip()
    save_point_cloud(filtered_pcd, base_filename, file_format="ply")

    o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")

    # Mesh creation
    print(filtered_pcd.get_max_bound())
    print(filtered_pcd.get_min_bound())

    print("Creating mesh using Ball Pivoting...")
    radii = [0.001, 0.005, 0.01]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        filtered_pcd,
        o3d.utility.DoubleVector(radii)
    )
    mesh.compute_vertex_normals()
    save_mesh(mesh, base_filename)

if __name__ == "__main__":
    main()
