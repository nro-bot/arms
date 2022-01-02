import time
import pickle
import numpy as np
import cv2
import constants
import paths
from camera import Camera
from robot.robot_arm import RobotArm
import utils


def main():

    c = Camera()
    c.start()
    c.load_calibration(paths.DEFAULT_CALIBRATION_PATH)
    time.sleep(1)

    with open(paths.DEFAULT_WORKSPACE_CALIBRATION_PATH, "rb") as f:
        d = pickle.load(f)

    robot2world = d["robot2world"]

    robot_points = np.array([
        [constants.WORKSPACE[0, 0], constants.WORKSPACE[1, 0]],
        [constants.WORKSPACE[0, 1], constants.WORKSPACE[1, 0]],
        [constants.WORKSPACE[0, 0], constants.WORKSPACE[1, 1]],
        [constants.WORKSPACE[0, 1], constants.WORKSPACE[1, 1]]
    ], dtype=np.float32)
    cam_points = utils.robot2world2cam_plane(robot_points, robot2world, c.world2cam, c.color_camera_matrix)
    print(cam_points)

    _, img = utils.realse_frame_to_numpy(c.get_frame())
    img = utils.draw_workspace(img, cam_points)
    utils.update_opencv_window(img)

    r = RobotArm()
    r.home()
    r.close_gripper()

    for idx, point in enumerate(robot_points):
        print("Moving to", point)
        r.move_hand_to([point[0], point[1], constants.Z_UP])
        r.move_hand_to([point[0], point[1], constants.Z_DOWN])
        time.sleep(0.2)
        _, img = utils.realse_frame_to_numpy(c.get_frame())
        img = utils.draw_workspace(img, cam_points)
        cv2.imwrite("data/workspace_{:d}.jpg".format(idx), img)
        utils.update_opencv_window(img)
        time.sleep(1)
        r.move_hand_to([point[0], point[1], constants.Z_UP])

    r.home()


main()
