import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

def capture_raw_frames(num_frames, save_dir="raw_frames"):
    """
    Captures raw depth and color frames from an Intel RealSense camera and saves them.
    - Depth frames are saved in both .npy (raw 16-bit) and .png (normalized 8-bit) formats.
    - Color frames are saved in both .npy and .png formats.
    """
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Configure the Intel RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 16-bit depth stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 8-bit color stream
    
    # Start streaming
    pipeline.start(config)
    
    try:
        for i in range(num_frames):
            print(f"\nCapturing frame {i + 1}...")
            
            # Wait for frames to be captured
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                print("Warning: Frames not captured correctly, skipping this frame.")
                continue
            
            # Convert depth and color frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())  # dtype: uint16 (16-bit depth)
            color_image = np.asanyarray(color_frame.get_data())  # dtype: uint8 (RGB image)
            
            # Normalize depth image for visualization (convert to 8-bit)
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)  # Convert to uint8 for saving as PNG
            
            # Save depth images
            np.save(f"{save_dir}/hand1_depth_frame_{i+1}.npy", depth_image)  # Raw 16-bit depth
            cv2.imwrite(f"{save_dir}/hand1_depth_frame_{i+1}.png", depth_normalized)  # 8-bit visualization
            
            # Save color images
            cv2.imwrite(f"{save_dir}/hand1_color_frame_{i+1}.png", color_image)  # Standard 8-bit color image
            np.save(f"{save_dir}/hand1_color_frame_{i+1}.npy", color_image)  # Raw color data
            
            print(f"Saved depth and color images for frame {i+1}.")
    
    finally:
        # Stop streaming when done
        pipeline.stop()

def main():
    """Main function to capture frames based on user input."""
    num_frames = int(input("Enter number of frames to capture: "))
    print("You have 5 seconds to position your hand...")
    time.sleep(5)  # Allow time for user to position their hand
    capture_raw_frames(num_frames)

if __name__ == "__main__":
    main()
