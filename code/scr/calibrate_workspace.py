import pickle
import time
import cv2
import numpy as np
from robot.robot_arm import RobotArm
from camera import Camera
import paths
import constants
import utils


def get_world_coordinate(c: Camera):

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
        cv2.imshow("RealSense", color)
        cv2.waitKey(1)
        time.sleep(1. / 30.)

    assert world_point is not None
    return world_point


def main():

    print("Click on the pixel where the robot's finger tip is touching the ground.")
    robot_points = np.array([
        [constants.WORKSPACE[0, 0], constants.WORKSPACE[1, 0]],
        [constants.WORKSPACE[0, 1], constants.WORKSPACE[1, 0]],
        [constants.WORKSPACE[0, 1], constants.WORKSPACE[1, 1]]
    ], dtype=np.float32)
    world_points = []

    c = Camera()
    c.start()
    c.load_calibration(paths.DEFAULT_CALIBRATION_PATH)

    r = RobotArm()
    r.home()
    r.close_gripper()

    for point in robot_points:
        r.move_hand_to([point[0], point[1], constants.Z_UP])
        r.move_hand_to([point[0], point[1], constants.Z_DOWN])
        world_points.append(get_world_coordinate(c))
        r.move_hand_to([point[0], point[1], constants.Z_UP])

    r.home()
    world_points = np.array(world_points, dtype=np.float32)

    # world2robot, robot2world
    # this works only for moving in the 2D plane of the workspace
    # i.e. I didn't calibrate the robot z-axis with the world z-axis
    world2robot = cv2.getAffineTransform(world_points, robot_points)
    robot2world = cv2.invertAffineTransform(world2robot)
    d = {
        "world2robot": world2robot,
        "robot2world": robot2world
    }

    with open(paths.DEFAULT_WORKSPACE_CALIBRATION_PATH, "wb") as f:
        pickle.dump(d, f)


main()
