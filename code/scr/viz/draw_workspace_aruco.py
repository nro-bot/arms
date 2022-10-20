import time
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
    time.sleep(1)
    c.load_calibration(paths.DEFAULT_WORKSPACE_CALIBRATION_PATH)

    world_points = np.array([
        [constants.WORKSPACE[0, 0], constants.WORKSPACE[1, 0]],
        [constants.WORKSPACE[0, 1], constants.WORKSPACE[1, 0]],
        [constants.WORKSPACE[0, 0], constants.WORKSPACE[1, 1]],
        [constants.WORKSPACE[0, 1], constants.WORKSPACE[1, 1]]
    ], dtype=np.float32)
    world_points = np.concatenate(
        [world_points, np.zeros((world_points.shape[0], 1), dtype=np.float32)],
        axis=1
    )
    print(world_points)
    cam_points = utils.coord_transform(c.world2cam, world_points)
    cam_points = np.dot(c.color_camera_matrix, cam_points.T).T
    cam_points[:, :2] /= cam_points[:, 2: 3]
    print(cam_points)

    _, img = utils.realse_frame_to_numpy(c.get_frame())
    img = utils.draw_workspace(img, cam_points)
    utils.update_opencv_window(img)

    r = RobotArm()
    for idx, point in enumerate(world_points):
        print("Moving to", point)
        r.move_hand_to([point[0], point[1], constants.Z_DOWN])
        time.sleep(0.2)
        _, img = utils.realse_frame_to_numpy(c.get_frame())
        img = utils.draw_workspace(img, cam_points)
        cv2.imwrite("data/workspace_{:d}.jpg".format(idx), img)
        utils.update_opencv_window(img)
        time.sleep(1)


main()
