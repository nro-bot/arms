import time
import pickle
import os
import numpy as np
import cv2
import constants
import paths
from camera import Camera
from robot.robot_arm import RobotArm
import utils


def get_world_coordinate(c: Camera, cam_workspace: np.ndarray):

    world_point = np.zeros(2, dtype=np.float32) - 1.

    def click_cam2world(event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        l = utils.pixel_to_cam_unknown_z(x, y, c.focal_length, c.principal_points)
        p = utils.calculate_intersection_with_ground(l, c.cam2world)

        if p is None:
            # no intersection with ground
            print("Ray does not intersect ground.")
            return

        world_point[:] = p[:2]

    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("RealSense", click_cam2world)

    while np.all(world_point == -1.):

        _, color = utils.realse_frame_to_numpy(c.get_frame())
        utils.draw_workspace(color, cam_workspace)
        cv2.imshow("RealSense", color)
        cv2.waitKey(1)
        time.sleep(1. / 30.)

    assert world_point is not None
    return world_point


def main():

    if not os.path.isfile(paths.DEFAULT_CALIBRATION_PATH):
        print("Run 'python -m scr.calibrate_camera'.")

    if not os.path.isfile(paths.DEFAULT_WORKSPACE_CALIBRATION_PATH):
        print("Run 'python -m scr.calibrate_workspace'.")

    with open(paths.DEFAULT_WORKSPACE_CALIBRATION_PATH, "rb") as f:
        d = pickle.load(f)

    robot2world = d["robot2world"]
    world2robot = d["world2robot"]

    c = Camera()
    c.start()
    time.sleep(1)
    c.load_calibration(paths.DEFAULT_CALIBRATION_PATH)

    robot_workspace = np.array([
        [constants.WORKSPACE[0, 0], constants.WORKSPACE[1, 0]],
        [constants.WORKSPACE[0, 1], constants.WORKSPACE[1, 0]],
        [constants.WORKSPACE[0, 0], constants.WORKSPACE[1, 1]],
        [constants.WORKSPACE[0, 1], constants.WORKSPACE[1, 1]]
    ], dtype=np.float32)
    cam_workspace = utils.robot2world2cam_plane(robot_workspace, robot2world ,c.world2cam, c.color_camera_matrix)

    r = RobotArm()
    r.home()
    r.open_gripper()

    while True:

        world_point = get_world_coordinate(c, cam_workspace)
        robot_point = utils.coord_transform_affine(world2robot, world_point)[0]

        r.open_gripper()
        r.move_hand_to([robot_point[0], robot_point[1], constants.Z_UP])
        r.move_hand_to([robot_point[0], robot_point[1], constants.Z_DOWN])
        r.close_gripper()
        r.move_hand_to([robot_point[0], robot_point[1], constants.Z_UP])


main()
