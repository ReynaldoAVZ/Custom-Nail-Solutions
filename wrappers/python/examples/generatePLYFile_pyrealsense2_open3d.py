import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
def capture_point_clouds(num_frames):
    """Captures multiple frames and returns a list of point clouds."""
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

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Debugging statements
            print(f"Depth Image Shape: {depth_image.shape}")  # Should be (480, 640)
            print(f"Color Image Shape: {color_image.shape}")  # Should be (480, 640, 3)
            
            if color_image.shape[-1] != 3:
                print("Error: Color image does not have 3 channels! Skipping frame.")
                continue

            # Convert depth to 3D point cloud
            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            points = []
            colors = []

            for y in range(depth_image.shape[0]):
                for x in range(depth_image.shape[1]):
                    depth_value = depth_image[y, x]
                    if depth_value == 0:  # Skip invalid points
                        continue

                    # Convert pixel to 3D coordinates
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
                    points.append(depth_point)
                    colors.append(color_image[y, x])  # Store BGR color

            # Convert to Open3D format
            points = np.array(points)
            colors = np.array(colors)

            print(f"Points Shape: {points.shape}")  # Should be (N, 3)
            print(f"Colors Shape: {colors.shape}")  # Should be (N, 3)

            if colors.shape[1] == 3:
                colors = colors.astype(np.float64) / 255.0  # Normalize to [0,1]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                captured_pcds.append(pcd)
            else:
                print("Error: Colors array does not have 3 channels! Skipping frame.")

            print("Capture complete!")

    finally:
        pipeline.stop()
    
    return captured_pcds

def apply_filters(pcd, enable_voxel_downsampling=True, enable_outlier_removal=True):
    """Applies filters to the point cloud (toggle on/off using parameters)."""
    print("\nApplying filters...")

    # Voxel Downsampling: Reduces point cloud sze, smooths data by clumping points together.
    # Tunable Parameter: voxel_size (meters)
    # Tuning Tips: 
    #   * Small value -> detailed pointcloud
    #   * Large value -> simpler pointcloud
    if enable_voxel_downsampling:
        print("Voxel downsampling enabled.")
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

    # Statistical Outlier Removal: Removes noise & isolated points
    # Tunable Parameter: nb_neighbors (number of surrounding points used as reference), std_ratio (standard deviation)
    # Tuning Tips: 
    #   * Higher nb_neighbors -> stronger filtering
    #   * Lower std_ratio -> stricter removal
    if enable_outlier_removal:
        print("Outlier removal enabled.")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    print("Filters applied!")
    return pcd

def align_point_clouds(pcds):
    """Aligns multiple point clouds using ICP (Iterative Closest Point)."""
    print("\nAligning point clouds using ICP...")

    if len(pcds) < 2:
        print("Warning: Not enough frames to align. Skipping alignment.")
        return pcds[0]

    reference_pcd = pcds[0]
    for i in range(1, len(pcds)):
        print(f"Aligning frame {i}...")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcds[i], reference_pcd, 0.02,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        pcds[i].transform(reg_p2p.transformation)
        reference_pcd += pcds[i]  # Merge aligned point clouds

    print("Alignment complete!")
    return reference_pcd

def save_point_cloud(pcd, filename="output9.ply"):
    """Saves the point cloud to a .ply file."""
    print(f"\nSaving final point cloud to {filename}...")
    o3d.io.write_point_cloud(filename, pcd)
    print("Save complete!")

def main():
    num_frames = int(input("Enter number of frames to capture: "))
    time.sleep(3) # provide some time to set hand and position
    captured_pcds = capture_point_clouds(num_frames)

    if not captured_pcds:
        print("Error: No point clouds were captured.")
        return

    # Align captured frames
    merged_pcd = align_point_clouds(captured_pcds)

    # Apply filters (toggle settings here)
    final_pcd = apply_filters(merged_pcd, enable_voxel_downsampling=True, enable_outlier_removal=True)

    # Save to file
    save_point_cloud(final_pcd)

if __name__ == "__main__":
    main()
