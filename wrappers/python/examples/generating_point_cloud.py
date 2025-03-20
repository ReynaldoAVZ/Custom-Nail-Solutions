import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream depth
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Create a pointcloud object
pc = rs.pointcloud()

# Create a frame align object
align = rs.align(rs.stream.color)

try:
    while True:
        # Wait for a coherent pair of frames: depth
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        if not depth_frame:
            continue

        # Generate the point cloud
        pc.map_to(depth_frame)
        points = pc.calculate(depth_frame)

        # Convert to numpy array and ensure the correct data type
        vtx = np.asanyarray(points.get_vertices())
        vtx = np.array([(p[0], p[1], p[2]) for p in vtx], dtype=np.float32)

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

finally:
    # Stop streaming
    pipeline.stop()