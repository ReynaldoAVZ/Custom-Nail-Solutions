import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import time

def get_o3d_intrinsics(profile):
    """
    Extract the intrinsics from the RealSense color stream and convert them to an Open3D PinholeCameraIntrinsic.
    """
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrin = color_profile.get_intrinsics()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        color_intrin.width, color_intrin.height,
        color_intrin.fx, color_intrin.fy,
        color_intrin.ppx, color_intrin.ppy)
    return intrinsic

def get_camera_pose(theta, distance):
    """
    Compute an extrinsic transformation matrix that places the camera on a circle of radius `distance`
    about the origin (assumed to be the center of your finger/nail). The camera is oriented to look
    toward the origin. 'theta' is the horizontal rotation angle (in radians).
    """
    # Camera position: moves on a circle around the origin.
    eye = np.array([distance * np.sin(theta), 0, distance * np.cos(theta)])
    center = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    
    # Compute the "look-at" axes. 
    # Note: in many computer vision libraries, the camera z-axis is defined as pointing *backwards* from the viewing direction.
    #  compute a matrix that transforms from the camera coordinate system to the world coordinate system.
    z = (eye - center)
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    
    # rotation matrix and translation vector.
    R = np.column_stack((x, y, z))
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = eye
    return extrinsic

def main():
    # RealSense pipeline.
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming.
    print("Starting camera...")
    pipeline_profile = pipeline.start(config)
    
    # sensor warm-up time.
    for i in range(10):
        pipeline.wait_for_frames()

    # Get the camera intrinsics from the color stream (for RGBD conversion).
    intrinsic = get_o3d_intrinsics(pipeline_profile)
    
    # Set up TSDF volume parameters.
    # Adjust voxel_length (resolution) and sdf_trunc based on the desired accuracy and scanned volume size.
    voxel_length = 0.0005  # voxel size in meters (0.5 mm)
    sdf_trunc = voxel_length * 10  # typically set to 10x the voxel_length
    tsdf_volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    
    # Parameters for the simulated rotation.
    num_frames = 12     # Number of frames to capture
    distance = 0.1      # Distance from the object in meters)
    
    try:
        print("Capturing frames and integrating into TSDF volume...")
        for i in range(num_frames):
            # Capture a set of frames.
            frames = pipeline.wait_for_frames()
            # If desired, you can use rs.align here to align depth to color.
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays.
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Convert the images to Open3D format.
            color_o3d = o3d.geometry.Image(color_image)
            depth_o3d = o3d.geometry.Image(depth_image)

            # Create an RGBD image.
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1000.0,  # adjust based on camera settings (often 1000 or 10000)
                depth_trunc=1.0,     # maximum depth value (meters)
                convert_rgb_to_intensity=False)
            
            # Simulate a camera pose: rotate from 0 to 180° over the captured frames.
            theta = (i / (num_frames - 1)) * np.pi  # theta in radians (0 to pi)
            extrinsic = get_camera_pose(theta, distance)
            
            # Note: TSDFVolume.integrate expects the transform from camera to world coordinates.
            # Since our get_camera_pose computes the world pose of the camera,
            # we need to pass its inverse.
            tsdf_volume.integrate(rgbd_image, intrinsic, np.linalg.inv(extrinsic))
            
            print(f"Integrated frame {i+1}/{num_frames} with theta={theta:.2f} rad")
            time.sleep(0.5)  # Optional: wait between captures.
    finally:
        # Stop the RealSense pipeline.
        pipeline.stop()
        print("Camera stopped.")

    # Extract a mesh from the integrated TSDF volume.
    print("Extracting mesh from TSDF volume (this may take a while)...")
    mesh = tsdf_volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # Visualize mesh.
    o3d.visualization.draw_geometries([mesh])
    # save mesh.
    o3d.io.write_triangle_mesh("nailbed_mesh.stl", mesh)
    print("Mesh saved as 'nailbed_mesh.stl'.")

if __name__ == '__main__':
    main()
