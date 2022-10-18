import argparse
import time
from camera import Camera
import paths
import exceptions
import utils
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np


def show_interactive(world_vertices):

    print("Interactive plot")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_vertices)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    d1_ax = fig.add_axes([0.25, 0.3, 0.65, 0.03])
    d2_ax = fig.add_axes([0.25, 0.25, 0.65, 0.03])
    d3_ax = fig.add_axes([0.25, 0.2, 0.65, 0.03])
    d4_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    d5_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    d6_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])

    smin, smax = -1., 1.
    xmin_s = Slider(d1_ax, 'xmin', smin, smax, valinit=-1)
    xmax_s = Slider(d2_ax, 'xmax', smin, smax, valinit=1)
    ymin_s = Slider(d3_ax, 'ymin', smin, smax, valinit=-1)
    ymax_s = Slider(d4_ax, 'ymax', smin, smax, valinit=1)
    zmin_s = Slider(d5_ax, 'zmin', smin, smax, valinit=-1)
    zmax_s = Slider(d6_ax, 'zmax', smin, smax, valinit=1)

    def sliders_on_changed(val):

        tmp = world_vertices
        tmp = tmp[np.logical_and(tmp[:, 0] >= xmin_s.val, tmp[:, 0] <= xmax_s.val)]
        tmp = tmp[np.logical_and(tmp[:, 1] >= ymin_s.val, tmp[:, 1] <= ymax_s.val)]
        tmp = tmp[np.logical_and(tmp[:, 2] >= zmin_s.val, tmp[:, 2] <= zmax_s.val)]
        pcd.points = o3d.utility.Vector3dVector(tmp)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    xmin_s.on_changed(sliders_on_changed)
    xmax_s.on_changed(sliders_on_changed)
    ymin_s.on_changed(sliders_on_changed)
    ymax_s.on_changed(sliders_on_changed)
    zmin_s.on_changed(sliders_on_changed)
    zmax_s.on_changed(sliders_on_changed)

    vis.poll_events()
    vis.update_renderer()
    plt.show()


def main(args):

    c = Camera()
    c.start()
    load_path = paths.SCAN_3D_CALIBRATION_TEMPLATE.format(args.camera)
    c.load_calibration(load_path)
    c.setup_pointcloud()
    time.sleep(1)

    try:

        cam_vertices = c.get_pointcloud()
        world_vertices = utils.coord_transform(c.cam2world, cam_vertices)

        tmp = world_vertices
        tmp = tmp[np.logical_and(tmp[:, 0] >= -0.1, tmp[:, 0] <= 0.4)]
        tmp = tmp[np.logical_and(tmp[:, 1] >= -0.25, tmp[:, 1] <= 0.37)]
        tmp = tmp[np.logical_and(tmp[:, 2] >= 0.0, tmp[:, 2] <= 1.)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tmp)
        o3d.visualization.draw_geometries([pcd])

        # show_interactive(world_vertices)

    finally:

        # Stop streaming
        c.stop()


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--camera", default=0, type=int)
parser.add_argument("--show", default=False, action="store_true", help="Show intermediate steps of calibration.")
main(parser.parse_args())
