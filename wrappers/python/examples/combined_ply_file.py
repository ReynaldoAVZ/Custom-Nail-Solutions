import pyrealsense2 as rs
import numpy as np

# Initialize pipeline and configure streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,30)  # Lower frame rate to 15 FPS
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
pipeline.start(config)

# Initialize pointcloud object
pc = rs.pointcloud()

# Initialize save_to_ply object
ply = rs.save_to_ply('combined_pointcloud_Reynaldo1.ply')
ply.set_option(rs.save_to_ply.option_ply_binary, False)
ply.set_option(rs.save_to_ply.option_ply_normals, True)
ply.set_option(rs.save_to_ply.option_ignore_color, False)

# Initialize frame queue
frame_queue = rs.frame_queue(10)

# Accumulate point clouds
accumulated_points = []
accumulated_texcoords = []

try:
    count = 0
    for _ in range(16):  # Capture 50 frames
        count += 1
        print(count)
        frames = pipeline.wait_for_frames(timeout_ms=10000)  # Increase timeout to 10 seconds
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Calculate point cloud
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        # Get vertices and texture coordinates
        vertices = np.asanyarray(points.get_vertices())
        texcoords = np.asanyarray(points.get_texture_coordinates())

        accumulated_points.append(vertices)
        accumulated_texcoords.append(texcoords)

finally:
    pipeline.stop()

# Combine accumulated point clouds
combined_points = np.concatenate(accumulated_points, axis=0)
combined_texcoords = np.concatenate(accumulated_texcoords, axis=0)

# Verify data before exporting
if combined_points.size > 0 and combined_texcoords.size > 0:
    # Create a new pointcloud object for combined data
    combined_pc = rs.pointcloud()

    # Export to .ply file using save_to_ply
    ply.process(frames)
else:
    print("No valid point cloud data to export.")