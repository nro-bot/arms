import argparse
import time
import cv2
import numpy as np
from robot.robot_arm import RobotArm
from camera import Camera
import exceptions
import paths
import constants
import utils


def get_corners(c):

    for i in range(5):

        _, color = utils.realse_frame_to_numpy(c.get_frame())
        ids, corners = c.detect_aruco_markers(color)

        utils.update_opencv_window(color)

        if ids is None:
            time.sleep(0.2)
            continue

        return corners[0][0]

    return None


def capture_points(c, r, workspace, z_up, z_down, open_close):

    r.home()
    if open_close:
        r.open_gripper()
        r.close_gripper()

    grid = utils.sample_points_in_workspace(workspace)
    found_points = []
    found_pixels = []

    for point in grid:

        r.move_hand_to([point[0], point[1], z_up])
        r.move_hand_to([point[0], point[1], z_down])

        time.sleep(0.2)
        tmp = get_corners(c)
        if tmp is not None:
            print(point, "found.")
            found_points.append(point)
            found_pixels.append(tmp)
        else:
            print(point, "not found.")

    r.home()
    return np.array(found_points, dtype=np.float32), np.array(found_pixels, dtype=np.float32)


def calculate_transform(found_points, found_pixels, c):

    found_points = np.concatenate(
        [found_points, np.zeros((found_points.shape[0], 1), dtype=np.float32)],
        axis=1
    )

    ret, rvec, tvec = cv2.solvePnP(found_points, found_pixels, c.color_camera_matrix, c.color_distortion_coeffs)

    if not ret:
        raise exceptions.CalibrationFailedException()

    world2cam = utils.transformation_matrix(rvec, tvec)
    cam2world = utils.inverse_transformation_matrix(rvec, tvec)

    return tvec, rvec, world2cam, cam2world


def main(args):

    c = Camera()
    c.start()
    c.setup_aruco()

    r = RobotArm()
    found_points, found_pixels = capture_points(c, r, constants.WORKSPACE, constants.Z_UP, constants.Z_DOWN, args.g)
    tvec, rvec, world2cam, cam2world = calculate_transform(found_points, found_pixels, c)

    c.set_calibration(tvec, rvec, world2cam, cam2world)
    c.save_calibration(paths.DEFAULT_WORKSPACE_CALIBRATION_PATH)
    print("Calibration data saved to {:s}.".format(paths.DEFAULT_WORKSPACE_CALIBRATION_PATH))


parser = argparse.ArgumentParser()
parser.add_argument("-g", default=False, action="store_true", help="open and close gripper at start")
main(parser.parse_args())
