# Written by: Reynaldo Villarreal Zambrano
# Optimized script for capturing and saving a colored point cloud from RealSense

# Import libraries
import pyrealsense2 as rs
import numpy as np

# Initialize pipeline and configure streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)

# Initialize pointcloud and save_to_ply object
pc = rs.pointcloud()
ply = rs.save_to_ply('combined_pointcloud_Reynaldo4.ply')
ply.set_option(rs.save_to_ply.option_ply_binary, False)  # Ensure human-readable PLY format
ply.set_option(rs.save_to_ply.option_ply_normals, True)  # Include normals for better visualization
ply.set_option(rs.save_to_ply.option_ignore_color, False)  # Ensure color data is saved

# Apply filtering to improve point cloud accuracy
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 50)

temporal = rs.temporal_filter()

# Initialize variables for frame averaging
averaged_points = None
frame_count = 0

try:
    for count in range(2):  # Capture multiple frames for a cleaner point cloud
        print(f"Capturing frame {count+1}")

        frames = pipeline.wait_for_frames(timeout_ms=5000)  # Increase timeout
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Skipping frame due to missing data")
            continue

        # Apply filters to depth data first
        # filtered_depth = spatial.process(depth_frame)
        # filtered_depth = temporal.process(filtered_depth)

        # Now map the color frame AFTER filtering depth data
        pc.map_to(color_frame)  # Ensure alignment after filtering
        points = pc.calculate(depth_frame)
        # points = pc.calculate(filtered_depth)  # Generate point cloud

        # Convert structured buffer data to a standard NumPy array
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        # Perform frame averaging
        if averaged_points is None:
            averaged_points = vertices
        else:
            averaged_points = (averaged_points * frame_count + vertices) / (frame_count + 1)

        frame_count += 1

        # Save frame incrementally to prevent memory overflow
        ply.process(points)

finally:
    pipeline.stop()
    print("Capture complete! Optimized colored point cloud saved.")
