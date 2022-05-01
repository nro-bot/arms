import numpy as np
import open3d as o3d
import utils


zipfile = np.load("data/plane1/pointcloud.npz")
points = zipfile["points.npy"]
pcd = utils.create_open3d_pointcloud(points, None)
o3d.visualization.draw_geometries([pcd])
