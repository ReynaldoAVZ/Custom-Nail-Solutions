import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time

def capture_point_clouds(num_frames):
    """
    Captures point clouds from a RealSense camera.
    Returns a list of Open3D point clouds.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    captured_pcds = []

    try:
        for i in range(num_frames):
            print(f"\nCapturing frame {i + 1}...")
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("Warning: Frames not captured correctly, skipping this frame.")
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if color_image.shape[-1] != 3:
                print("Error: Color image does not have 3 channels! Skipping frame.")
                continue

            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            points = []
            colors = []

            # Convert each depth pixel to 3D coordinates
            for y in range(depth_image.shape[0]):
                for x in range(depth_image.shape[1]):
                    depth_value = depth_image[y, x]
                    if depth_value == 0:
                        continue
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
                    depth_point[2] *= -1  # Flip Z
                    points.append(depth_point)
                    colors.append(color_image[y, x])  # BGR format

            points = np.array(points)
            colors = np.array(colors)

            if colors.shape[1] == 3:
                colors = colors.astype(np.float64) / 255.0  # Normalize to [0, 1]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                captured_pcds.append(pcd)
                print("Frame captured successfully!")
            else:
                print("Error: Invalid color shape. Skipping frame.")

    finally:
        pipeline.stop()

    return captured_pcds

def apply_filters(pcd, enable_voxel_downsampling=True, enable_outlier_removal=True):
    """
    Applies optional voxel downsampling and statistical outlier removal to the point cloud.
    """
    print("\nApplying filters...")

    if enable_voxel_downsampling:
        print(" - Applying voxel downsampling...")
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

    if enable_outlier_removal:
        print(" - Removing outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    print("Filtering complete.")
    return pcd

def align_point_clouds(pcds):
    """
    Aligns a list of point clouds using pairwise ICP registration.
    Returns a merged, aligned point cloud.
    """
    print("\nAligning point clouds using ICP...")

    if len(pcds) < 2:
        print("Warning: Not enough point clouds for alignment.")
        return pcds[0]

    aligned_pcd = pcds[0]

    for i in range(1, len(pcds)):
        print(f" - Aligning frame {i}...")

        # Rough initial alignment by center offset
        prev_center = np.mean(np.asarray(aligned_pcd.points), axis=0)
        curr_center = np.mean(np.asarray(pcds[i].points), axis=0)
        init_guess = np.eye(4)
        init_guess[:3, 3] = prev_center - curr_center

        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcds[i], aligned_pcd, 0.02, init_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        pcds[i].transform(reg_p2p.transformation)
        aligned_pcd += pcds[i]

    print("ICP alignment complete.")
    return aligned_pcd

def save_point_cloud(pcd, filename):
    """
    Saves an Open3D point cloud as a PLY file.
    """
    print(f"\nSaving point cloud to '{filename}.ply'...")
    o3d.io.write_point_cloud(f"{filename}.ply", pcd)
    print("Point cloud saved.")

def create_mesh_from_pcd(pcd, method="poisson"):
    """
    Generates a 3D mesh from a point cloud.
    Supports 'poisson' or 'ball_pivot' methods.
    """
    print("\nEstimating normals for meshing...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    if method == "poisson":
        print("Creating mesh using Poisson surface reconstruction...")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    elif method == "ball_pivot":
        print("Creating mesh using Ball Pivoting algorithm...")
        dists = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(dists)
        radius = 3 * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )
    else:
        raise ValueError("Unsupported mesh method. Use 'poisson' or 'ball_pivot'.")

    print("Mesh creation complete.")
    return mesh

def save_mesh(mesh, filename):
    """
    Saves a mesh to an STL file.
    """
    print(f"\nSaving mesh to '{filename}.stl'...")
    o3d.io.write_triangle_mesh(f"{filename}.stl", mesh)
    print("Mesh saved.")

def main():
    """
    Main pipeline:
    1. Ask for number of frames and output name
    2. Capture frames
    3. Align and filter point clouds
    4. Save point cloud as .ply
    5. Create mesh and save as .stl
    """
    num_frames = int(input("Enter number of frames to capture: "))
    filename = input("Enter base filename (without extension) for saving output: ").strip()

    print("You have 3 seconds to place your hand/scene...")
    time.sleep(3)

    captured_pcds = capture_point_clouds(num_frames)
    if not captured_pcds:
        print("No point clouds captured. Exiting.")
        return

    merged_pcd = align_point_clouds(captured_pcds)
    final_pcd = apply_filters(merged_pcd)

    save_point_cloud(final_pcd, filename=filename)

    mesh = create_mesh_from_pcd(final_pcd, method="poisson")
    save_mesh(mesh, filename=filename)

    print("\nDone! You can now view the .ply or .stl file in MeshLab or any 3D viewer.")
    o3d.visualization.draw_geometries([mesh], window_name="Final Mesh")

if __name__ == "__main__":
    main()
