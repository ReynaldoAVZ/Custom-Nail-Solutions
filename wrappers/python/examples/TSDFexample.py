#TSDF Example
#https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html
# %% Activation
'''In the activation step, we first locate blocks that contain points unprojected 
from the current depth image. In other words, it finds active blocks in the current 
viewing frustum. Internally, this is achieved by a frustum hash map that produces 
duplication-free block coordinates, and a block hash map that activates and query such 
block coordinates.'''
# examples/python/t_reconstruction_system/integrate.py
frustum_block_coords = vbg.compute_unique_block_coordinates(
    depth, depth_intrinsic, extrinsic, config.depth_scale,
    config.depth_max)

# %% Integration
'''Now we can process the voxels in the blocks at frustum_block_coords. This is done by 
projecting all such related voxels to the input images and perform a weighted average, 
which is a pure geometric process without hash map operations.
We may use optimized functions, along with raw depth images with calibration parameters 
to activate and perform TSDF integration, optionally with colors:'''
# examples/python/t_reconstruction_system/integrate.py
if config.integrate_color:
    color = o3d.t.io.read_image(color_file_names[i]).to(device)
    vbg.integrate(frustum_block_coords, depth, color, depth_intrinsic,
                    color_intrinsic, extrinsic, config.depth_scale,
                    config.depth_max)
else:
    vbg.integrate(frustum_block_coords, depth, depth_intrinsic,
                    extrinsic, config.depth_scale, config.depth_max)

# %% Surface extraction
# examples/python/t_reconstruction_system/integrate.py
# You may use the provided APIs to extract surface points.
pcd = vbg.extract_point_cloud()
o3d.visualization.draw([pcd])

mesh = vbg.extract_triangle_mesh()
o3d.visualization.draw([mesh.to_legacy()])