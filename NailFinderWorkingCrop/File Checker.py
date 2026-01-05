import open3d as o3d
import numpy as np
import tkinter as tk 
from tkinter import filedialog 
import os

def check_point_cloud_scale():
    """
    Prompts the user to select a PLY file and analyzes its spatial dimensions 
    (min/max X, Y, Z coordinates) to verify the units of measurement.
    """
    # Initialize tkinter for file dialog
    root = tk.Tk()
    root.withdraw() 
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select a Single PLY Scan File to Check Scale",
        filetypes=[("PLY files", "*.ply")]
    )
    
    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Loading file: {os.path.basename(file_path)}")
    
    try:
        # Read the point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        
        if not pcd.has_points():
            print("ERROR: The selected file does not contain any points.")
            return

        # Get the 3D coordinates (points) as a NumPy array
        points = np.asarray(pcd.points)

        # Calculate the bounding box statistics
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Calculate the dimensions (length, width, height)
        dimensions = max_coords - min_coords

        print("\n--- Point Cloud Scale Analysis ---")
        
        # Print minimum coordinates (often close to the camera or origin)
        print("Minimum Coordinates (X_min, Y_min, Z_min):")
        print(f"  X: {min_coords[0]:.4f}")
        print(f"  Y: {min_coords[1]:.4f}")
        print(f"  Z: {min_coords[2]:.4f} <-- This Z value is the closest distance to the camera.")

        print("\nMaximum Coordinates (X_max, Y_max, Z_max):")
        print(f"  X: {max_coords[0]:.4f}")
        print(f"  Y: {max_coords[1]:.4f}")
        print(f"  Z: {max_coords[2]:.4f} <-- This Z value is the farthest distance from the camera.")

        print("\nSpatial Dimensions (Width, Height, Depth):")
        print(f"  Width (X span): {dimensions[0]:.4f}")
        print(f"  Height (Y span): {dimensions[1]:.4f}")
        print(f"  Depth (Z span): {dimensions[2]:.4f}")
        print("----------------------------------")

        # Interpret the likely units
        # If Z values are ~0.09 and dimensions are ~0.1, units are meters (m).
        # If Z values are ~90 and dimensions are ~100, units are millimeters (mm).
        # If Z values are ~9 and dimensions are ~10, units are centimeters (cm).
        
        if (min_coords[2] > 0.05 and min_coords[2] < 0.5):
            print("INTERPRETATION: Based on the Z-values, the units are highly likely to be **METERS (m)**.")
            print(f"Your closest point is {min_coords[2]*100:.1f} cm away (approx {min_coords[2]/0.0254:.1f} inches).")
        elif (min_coords[2] > 5 and min_coords[2] < 50):
            print("INTERPRETATION: Based on the Z-values, the units are likely to be **CENTIMETERS (cm)**.")
            print(f"Your closest point is {min_coords[2]:.1f} cm away (approx {min_coords[2]/2.54:.1f} inches).")
        elif (min_coords[2] > 50 and min_coords[2] < 500):
            print("INTERPRETATION: Based on the Z-values, the units are likely to be **MILLIMETERS (mm)**.")
            print(f"Your closest point is {min_coords[2]/10:.1f} cm away (approx {min_coords[2]/25.4:.1f} inches).")
        else:
            print("INTERPRETATION: Units are ambiguous. Please analyze the printed values against your known dimensions (9 cm distance, 10 cm hand width).")


    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    check_point_cloud_scale()
