import open3d as o3d
import numpy as np
from tkinter import Tk, filedialog
import os


def select_file():
    """
    Opens a file dialog for the user to select a point cloud file.
    Supported formats include .ply, .pcd, etc.
    """
    print("Select a point cloud file...")
    root = Tk()
    root.withdraw()  # Hide the Tkinter main window
    file_path = filedialog.askopenfilename(
        title="Select a Point Cloud File",
        filetypes=[("Point Cloud Files", "*.ply *.pcd *.xyz *.pts"), ("All Files", "*.*")]
    )
    if not file_path:
        raise ValueError("No file selected!")
    print(f"Selected file: {file_path}")
    return file_path


def load_point_cloud(filename):
    """
    Loads a point cloud from a given file.
    Supported formats include .ply, .pcd, etc.
    """
    print(f"Loading point cloud from '{filename}'...")
    pcd = o3d.io.read_point_cloud(filename)
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    return pcd


def apply_filters(pcd, enable_voxel_downsampling=True, enable_outlier_removal=True):
    """
    Cleans the point cloud by applying voxel downsampling and outlier removal.
    """
    print("\nApplying filters...")
    if enable_voxel_downsampling:
        print(" - Applying voxel downsampling...")
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

    if enable_outlier_removal:
        print(" - Removing statistical outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    print("Filtering complete.")
    return pcd


def remove_outliers(pcd, radius=0.02, min_neighbors=16):
    """
    Removes outliers using radius-based filtering.
    """
    print(f"Removing outliers with radius={radius} and min_neighbors={min_neighbors}...")
    pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    print("Radius-based outlier removal complete.")
    return pcd


def smooth_point_cloud(pcd, search_radius=0.01):
    """
    Smooths the point cloud using Moving Least Squares (MLS).
    """
    print("Smoothing point cloud...")
    smoothed_pcd = pcd.uniform_down_sample(every_k_points=1)
    smoothed_pcd = smoothed_pcd.voxel_down_sample(voxel_size=search_radius)
    print("Smoothing complete.")
    return smoothed_pcd


def compute_normals(pcd, radius=0.01, max_nn=30):
    """
    Estimates the normals of the point cloud.
    """
    print("\nEstimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(100)
    print("Normals estimated successfully.")
    return pcd


def create_mesh_from_pcd(pcd, method="poisson"):
    """
    Generates a mesh from the point cloud using Poisson or Ball Pivoting methods.
    """
    print("\nCreating mesh...")
    if method == "poisson":
        print(" - Using Poisson reconstruction...")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    elif method == "ball_pivot":
        print(" - Using Ball Pivoting algorithm...")
        dists = pcd.compute_nearest_neighbor_distance()
        radius = 3 * np.mean(dists)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )
    else:
        raise ValueError("Invalid mesh generation method.")

    print(" - Mesh creation complete.")
    
    # Compute and ensure normals are correctly set
    mesh.compute_vertex_normals()
    print(" - Mesh normals computed.")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    return mesh



def simplify_mesh(mesh, target_reduction=0.5):
    """
    Simplifies a mesh by reducing the number of triangles.
    target_reduction specifies the fraction of triangles to retain.
    """
    print(f"Simplifying mesh with target reduction: {target_reduction*100:.0f}%...")
    simplified_mesh = mesh.simplify_quadric_decimation(int(len(mesh.triangles) * target_reduction))
    print(f"Reduced from {len(mesh.triangles)} to {len(simplified_mesh.triangles)} triangles.")
    return simplified_mesh


def fill_mesh_holes(mesh):
    """
    Fills small holes in a mesh.
    """
    print("Filling holes in the mesh...")
    mesh.remove_unreferenced_vertices()
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    print("Hole filling complete.")
    return mesh


def save_mesh(mesh, filename):
    """
    Saves the mesh to a file in STL format.
    """
    print(f"\nSaving mesh to '{filename}.stl'...")
    o3d.io.write_triangle_mesh(f"{filename}.stl", mesh)
    print("Mesh saved successfully.")


def save_point_cloud(pcd, filename):
    """
    Saves the point cloud to a file in PLY format.
    """
    print(f"\nSaving filtered point cloud to '{filename}.ply'...")
    o3d.io.write_point_cloud(f"{filename}.ply", pcd)
    print("Point cloud saved successfully.")


def main():
    # Open file dialog to select a point cloud file
    input_filename = select_file()
    base_filename, _ = os.path.splitext(input_filename)

    # Load the point cloud
    pcd = load_point_cloud(input_filename)

    # Apply filters and clean the data
    filtered_pcd = apply_filters(pcd)
    filtered_pcd = remove_outliers(filtered_pcd, radius=0.02, min_neighbors=20)
    filtered_pcd = smooth_point_cloud(filtered_pcd, search_radius=0.01)

    # Compute normals for the point cloud
    filtered_pcd = compute_normals(filtered_pcd)

    # Save the cleaned point cloud for further use
    save_point_cloud(filtered_pcd, base_filename)

    # Generate a mesh from the point cloud
    mesh = create_mesh_from_pcd(filtered_pcd, method="poisson")

    # Simplify the mesh and fill holes
    mesh.compute_vertex_normals()  # Ensures normals are properly computed
    mesh = simplify_mesh(mesh, target_reduction=1)
    mesh = fill_mesh_holes(mesh)

    # Save the mesh as an STL file
    save_mesh(mesh, base_filename)

    # Visualize the processed mesh
    o3d.visualization.draw_geometries([mesh], window_name="Final Mesh")


if __name__ == "__main__":
    main()
