import pyrealsense2 as rs
import numpy as np

def save_intrinsics_to_file(camera_serial_number=None, filename="intrinsics.npy"):
    """
    Captures camera intrinsics from an Intel RealSense device and saves them to a file.
    """
    try:
        # Initialize a RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()

        # Optional: Set the serial number of the camera to use a specific device
        if camera_serial_number:
            config.enable_device(camera_serial_number)

        # Enable a depth stream (or any desired stream)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start the pipeline
        pipeline.start(config)

        # Get the active profile and extract intrinsics
        profile = pipeline.get_active_profile()
        depth_stream = profile.get_stream(rs.stream.depth)
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        # Create a dictionary to save the intrinsics
        intrinsics_dict = {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'ppx': intrinsics.ppx,
            'ppy': intrinsics.ppy,
            'coeffs': intrinsics.coeffs,
            'depth_scale': get_depth_scale(profile)  # Get depth scale from the device
        }
        print('width', intrinsics_dict['width'])
        print('height', intrinsics_dict['height']) 
        print('fx', intrinsics_dict['fx'])
        print('fy', intrinsics_dict['fy'])
        print('ppx', intrinsics_dict['ppx'])
        print('ppy', intrinsics_dict['ppy'])
        print('coeffs', intrinsics_dict['coeffs'])
        print('depth_scale', intrinsics_dict['depth_scale'])

        # Save the intrinsics to a .npy file
        np.save(filename, intrinsics_dict)
        print(f"Intrinsics successfully saved to {filename}")

    except Exception as e:
        print(f"Failed to capture intrinsics: {e}")

    finally:
        # Stop the pipeline
        pipeline.stop()

def get_depth_scale(profile):
    """
    Retrieves the depth scale (depth unit to meters conversion factor) from the device.
    """
    sensor = profile.get_device().first_depth_sensor()
    return sensor.get_depth_scale()

if __name__ == "__main__":
    save_intrinsics_to_file(filename="intrinsics.npy")
