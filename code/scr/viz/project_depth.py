import pickle
import time
import numpy as np
from camera import Camera
import utils
import constants
import paths
import pyximport; pyximport.install()  # this will compile cython code
import pyx.render as render

IMAGE_SIZE = 500
MIN_VALUE = 0  # 0 = on the ground
MAX_VALUE = 0.03  # 0.05 = 5cm above the ground


def main():
    # setup camera
    c = Camera()
    c.start()
    c.load_calibration(paths.DEFAULT_CALIBRATION_PATH)
    c.setup_pointcloud()

    # this tells us the robot's workspace
    # we only want to show objects inside of the workspace
    with open(paths.DEFAULT_WORKSPACE_CALIBRATION_PATH, "rb") as f:
        d = pickle.load(f)
    robot2world = d["robot2world"]
    world2robot = d["world2robot"]

    # TODO: here we assume the camera and robot frame aren't rotated w.r.t. each other
    robot_points = np.array([
        [constants.WORKSPACE[0, 0], constants.WORKSPACE[1, 0]],
        [constants.WORKSPACE[0, 1], constants.WORKSPACE[1, 1]]
    ], dtype=np.float32)
    workspace = utils.robot2world_plane(robot_points, robot2world)
    workspace = np.array([
        [workspace[0, 0], workspace[1, 0]],
        [workspace[1, 1], workspace[0, 1]]  # TODO: not sure what's going on here
    ])
    # workspace = np.array([
    #     [-0.2, 0.2],
    #     [-0.2, 0.2]
    # ], dtype=np.float32)

    # render top-down depth images inside of the robot's workspace
    while True:

        cam_vertices = c.get_pointcloud()
        world_vertices = utils.coord_transform(c.cam2world, cam_vertices)
        depth = utils.project_top_down_depth(world_vertices, workspace, IMAGE_SIZE, MIN_VALUE, MAX_VALUE)
        image = ((depth / MAX_VALUE) * 255).astype(np.uint8)
        utils.update_opencv_window(image)
        time.sleep(1 / 30)


main()
