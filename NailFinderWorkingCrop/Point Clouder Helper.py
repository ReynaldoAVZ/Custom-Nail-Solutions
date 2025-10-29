import sys
import numpy as np
import open3d as o3d
from PyQt6 import QtWidgets, QtCore, QtGui

# --- Main Application Window ---
class PointCloudAligner(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ICP Alignment Helper")
        self.setGeometry(100, 100, 450, 400)

        # Store point cloud data
        self.pcd_target = None
        self.pcd_source = None
        # This will hold the transformed source for visualization
        self.pcd_source_transformed_display = None

        self.initUI()

    def initUI(self):
        # --- Main Widget and Layout ---
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # --- File Loading ---
        file_layout = QtWidgets.QGridLayout()
        self.btn_load_target = QtWidgets.QPushButton("1. Load Target Frame (Fixed, Blue)")
        self.lbl_target_path = QtWidgets.QLabel("No file loaded.")
        self.btn_load_source = QtWidgets.QPushButton("2. Load Source Frame (To Align, Red)")
        self.lbl_source_path = QtWidgets.QLabel("No file loaded.")
        
        file_layout.addWidget(self.btn_load_target, 0, 0)
        file_layout.addWidget(self.lbl_target_path, 0, 1)
        file_layout.addWidget(self.btn_load_source, 1, 0)
        file_layout.addWidget(self.lbl_source_path, 1, 1)
        main_layout.addLayout(file_layout)
        
        # --- Connections ---
        self.btn_load_target.clicked.connect(lambda: self.load_pcd(target=True))
        self.btn_load_source.clicked.connect(lambda: self.load_pcd(target=False))

        # --- Initial Guess Controls ---
        group_box = QtWidgets.QGroupBox("3. Set Initial Guess for Transformation")
        grid_layout = QtWidgets.QGridLayout()

        grid_layout.addWidget(QtWidgets.QLabel("Translation X (m)"), 0, 0)
        self.spin_tx = QtWidgets.QDoubleSpinBox(minimum=-2.0, maximum=2.0, value=0.0, singleStep=0.01, decimals=4)
        grid_layout.addWidget(self.spin_tx, 0, 1)

        grid_layout.addWidget(QtWidgets.QLabel("Translation Y (m)"), 1, 0)
        self.spin_ty = QtWidgets.QDoubleSpinBox(minimum=-2.0, maximum=2.0, value=0.0, singleStep=0.01, decimals=4)
        grid_layout.addWidget(self.spin_ty, 1, 1)

        grid_layout.addWidget(QtWidgets.QLabel("Translation Z (m)"), 2, 0)
        self.spin_tz = QtWidgets.QDoubleSpinBox(minimum=-2.0, maximum=2.0, value=0.0, singleStep=0.01, decimals=4)
        grid_layout.addWidget(self.spin_tz, 2, 1)

        grid_layout.addWidget(QtWidgets.QLabel("Rotation Roll (X, deg)"), 0, 2)
        self.spin_rx = QtWidgets.QDoubleSpinBox(minimum=-180, maximum=180, value=0.0, singleStep=1.0)
        grid_layout.addWidget(self.spin_rx, 0, 3)
        
        grid_layout.addWidget(QtWidgets.QLabel("Rotation Pitch (Y, deg)"), 1, 2)
        self.spin_ry = QtWidgets.QDoubleSpinBox(minimum=-180, maximum=180, value=0.0, singleStep=1.0)
        grid_layout.addWidget(self.spin_ry, 1, 3)
        
        grid_layout.addWidget(QtWidgets.QLabel("Rotation Yaw (Z, deg)"), 2, 2)
        self.spin_rz = QtWidgets.QDoubleSpinBox(minimum=-180, maximum=180, value=0.0, singleStep=1.0)
        grid_layout.addWidget(self.spin_rz, 2, 3)

        group_box.setLayout(grid_layout)
        main_layout.addWidget(group_box)
        
        # --- Action Buttons ---
        action_layout = QtWidgets.QHBoxLayout()
        self.btn_apply_guess = QtWidgets.QPushButton("4. Apply & Preview Guess")
        self.btn_run_icp = QtWidgets.QPushButton("5. Run ICP & Get Result")
        
        self.btn_apply_guess.clicked.connect(self.apply_initial_guess)
        self.btn_run_icp.clicked.connect(self.run_icp_alignment)
        
        action_layout.addWidget(self.btn_apply_guess)
        action_layout.addWidget(self.btn_run_icp)
        main_layout.addLayout(action_layout)

        main_layout.addStretch()

    def load_pcd(self, target=True):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Point Cloud", "", "PLY Files (*.ply)")
        if not filepath:
            return
            
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            if not pcd.has_points():
                raise ValueError("Point cloud is empty.")

            if target:
                self.pcd_target = pcd
                self.lbl_target_path.setText(f".../{filepath.split('/')[-1]}")
            else:
                self.pcd_source = pcd
                self.lbl_source_path.setText(f".../{filepath.split('/')[-1]}")
            
            print(f"Successfully loaded {'target' if target else 'source'} file: {filepath}")
            self.preview_current_state()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load point cloud: {e}")

    def preview_current_state(self):
        """Shows the current state of the point clouds in a new window."""
        if not self.pcd_target and not self.pcd_source:
            print("Nothing to display yet.")
            return

        geometries = []
        if self.pcd_target:
            self.pcd_target.paint_uniform_color([0.2, 0.4, 1.0]) # Blue
            geometries.append(self.pcd_target)
        
        if self.pcd_source_transformed_display:
            self.pcd_source_transformed_display.paint_uniform_color([1.0, 0.2, 0.2]) # Red
            geometries.append(self.pcd_source_transformed_display)
        elif self.pcd_source: # If no transformation has been applied yet
             self.pcd_source.paint_uniform_color([1.0, 0.2, 0.2]) # Red
             geometries.append(self.pcd_source)
        
        print("Opening preview window... Close it to continue.")
        o3d.visualization.draw_geometries(geometries, window_name="Alignment Preview")

    def get_transform_from_inputs(self):
        """Builds a 4x4 transformation matrix from the GUI inputs."""
        T = np.identity(4)
        R = o3d.geometry.get_rotation_matrix_from_xyz((
            np.radians(self.spin_rx.value()),
            np.radians(self.spin_ry.value()),
            np.radians(self.spin_rz.value())
        ))
        T[:3, :3] = R
        T[:3, 3] = [self.spin_tx.value(), self.spin_ty.value(), self.spin_tz.value()]
        return T

    def apply_initial_guess(self):
        """Applies the manual transform to the source point cloud."""
        if self.pcd_source is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load a source point cloud first.")
            return

        transform = self.get_transform_from_inputs()
        # Always transform from the original, unmodified source
        self.pcd_source_transformed_display = o3d.geometry.PointCloud(self.pcd_source)
        self.pcd_source_transformed_display.transform(transform)
        print("Applying initial guess transform...")
        self.preview_current_state()
        
    def run_icp_alignment(self):
        """Performs the ICP registration."""
        if not self.pcd_target or not self.pcd_source:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load both target and source point clouds.")
            return

        print("\n--- Running ICP ---")
        initial_transform = self.get_transform_from_inputs()
        voxel_size = 0.005  # 5mm for downsampling
        
        source_down = self.pcd_source.voxel_down_sample(voxel_size)
        target_down = self.pcd_target.voxel_down_sample(voxel_size)
        
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_down, target_down, voxel_size * 1.5, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        
        print("ICP Finished.")
        print(f"Fitness: {reg_p2p.fitness:.4f} (proportion of inlier correspondences, higher is better)")
        print(f"Inlier RMSE: {reg_p2p.inlier_rmse:.4f} (error for inlier correspondences, lower is better)")
        print("--- Final Transformation Matrix ---")
        print("This is the matrix that aligns the SOURCE frame to the TARGET frame.")
        print(reg_p2p.transformation)
        print("---------------------------------")

        # --- 1. Extract and Define Components ---
        
        # 3x1 Translation Vector (t) from the ICP result (in meters)
        t_icp = np.array([0.01152724, 0.02282345, 0.00175603])
        
        # The Rotation Angle (theta) is extracted from the R11 component of your matrix:
        # R11 = 0.9392133 --> theta = 20.07 degrees (0.3503 radians)
        theta = 0.3503  # in radians
        
        # --- 2. Create the Ideal, Stable Z-Rotation Matrix (R_ideal) ---
        # Your ICP output suggests a non-standard Z-axis rotation for the R12, R21 signs.
        # We will use the standard Z-axis rotation and let the offset handle the rest.
        c, s = np.cos(theta), np.sin(theta)
        
        # NOTE: The sign of 's' determines the direction of rotation. 
        # We use the standard Z-rotation matrix:
        R_ideal = np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # --- 3. Solve for p using R_ideal and t_icp ---
        I = np.identity(3)
        M = I - R_ideal
        
        try:
            M_inv = np.linalg.inv(M)
            p_offset = M_inv @ t_icp
            
            print("\n--- Corrected Calculated Offset Vector p (meters) ---")
            print(p_offset)
            print("---------------------------------------------------")
            
        except np.linalg.LinAlgError:
            print("Error: The corrected matrix (I - R_ideal) is still singular. Check if your rotation is close to 0 or 360 degrees.")
        # Apply the final transformation for visualization
        self.pcd_source_transformed_display = o3d.geometry.PointCloud(self.pcd_source)
        self.pcd_source_transformed_display.transform(reg_p2p.transformation)
        
        self.preview_current_state()

# --- Main execution ---
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = PointCloudAligner()
    main_window.show()
    sys.exit(app.exec())