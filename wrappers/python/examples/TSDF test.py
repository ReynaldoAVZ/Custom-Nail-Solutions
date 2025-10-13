import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import time

def get_o3d_intrinsics(profile):
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrin = color_profile.get_intrinsics()
    return o3d.camera.PinholeCameraIntrinsic(
        color_intrin.width, color_intrin.height,
        color_intrin.fx, color_intrin.fy,
        color_intrin.ppx, color_intrin.ppy
    )

def get_camera_pose(theta, distance):
    eye = np.array([distance * np.sin(theta), 0, distance * np.cos(theta)])
    center = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    z = (eye - center)
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    R = np.column_stack((x, y, z))
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = eye
    return extrinsic

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("Starting camera...")
    pipeline_profile = pipeline.start(config)

    align = rs.align(rs.stream.color)  # Align depth to color
    for _ in range(10):  # Warm-up
        pipeline.wait_for_frames()

    intrinsic = get_o3d_intrinsics(pipeline_profile)

    voxel_length = 0.0005  # 0.5 mm resolution
    sdf_trunc = voxel_length * 10

    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    num_frames = 3
    distance = 0.1  # meters

    try:
        print("Capturing frames and integrating into TSDF volume...")
        for i in range(num_frames):
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Check if the depth frame has enough valid data
            if np.count_nonzero(depth_image) < 1000:
                print(f"[WARNING] Skipping frame {i+1} due to insufficient depth data.")
                continue

            print(f"Depth image shape: {depth_image.shape}, min: {np.min(depth_image)}, max: {np.max(depth_image)}")
            print(f"Color image shape: {color_image.shape}")

            color_o3d = o3d.geometry.Image(color_image)
            depth_o3d = o3d.geometry.Image(depth_image)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1000.0,
                depth_trunc=2.0,  # Increased max depth to 2m
                convert_rgb_to_intensity=False)

            theta = (i / (num_frames)) * np.pi  # 0 to pi
            theta = 0
            extrinsic = get_camera_pose(theta, distance)
            tsdf_volume.integrate(rgbd_image, intrinsic, np.linalg.inv(extrinsic))

            print(f"Integrated frame {i+1}/{num_frames} with theta={theta:.2f} rad")
            time.sleep(0.5)

    finally:
        pipeline.stop()
        print("Camera stopped.")

    print("Extracting mesh from TSDF volume (this may take a while)...")
    mesh = tsdf_volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0:
        print("[ERROR] Mesh contains 0 vertices. Likely no integration happened.")
    else:
        print(f"[INFO] Mesh extracted. Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("nailbed_mesh.stl", mesh)
    print("Mesh saved as 'nailbed_mesh2.stl'.")

if __name__ == '__main__':
    main()
