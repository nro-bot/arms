import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from camera import Camera
import utils
import paths


def main():

    c = Camera()
    c.start()
    c.load_calibration(paths.DEFAULT_CALIBRATION_PATH)

    pc = rs.pointcloud()

    # TODO: this should eventually be the robot's workspace
    # I'm testing this only with the camera, so I use some arbitrary values
    workspace = np.array([
        [-0.2, 0.2],
        [-0.2, 0.2]
    ], dtype=np.float32)

    while True:

        # create a point cloud
        depth_frame, color_frame = utils.realse_frame(c.get_frame())
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)

        tmp = np.asanyarray(points.get_vertices())
        cam_vertices = np.zeros((len(tmp), 3), dtype=np.float32)
        for i, t in enumerate(tmp):
            cam_vertices[i][0] = t[0]
            cam_vertices[i][1] = t[1]
            cam_vertices[i][2] = t[2]

        world_vertices = utils.coord_transform(c.cam2world, cam_vertices)

        not_all_zeros = np.any(world_vertices != 0, axis=1)
        world_vertices = world_vertices[not_all_zeros]

        mask = np.logical_and(world_vertices[:, 0] >= workspace[0, 0], world_vertices[:, 0] <= workspace[0, 1])
        mask = np.logical_and(mask, world_vertices[:, 1] >= workspace[1, 0])
        mask = np.logical_and(mask, world_vertices[:, 1] <= workspace[1, 1])

        print(world_vertices[:, 2].min(), world_vertices[:, 2].max(), world_vertices[:, 2].mean())

        captured_vertices = world_vertices[mask]
        captured_vertices[:, 0] -= workspace[0, 0]
        captured_vertices[:, 1] -= workspace[1, 0]
        captured_vertices[:, 0] /= (workspace[0, 1] - workspace[0, 0])
        captured_vertices[:, 1] /= (workspace[1, 1] - workspace[1, 0])
        captured_vertices[:, 0] *= 99
        captured_vertices[:, 1] *= 99
        captured_vertices[:, :2] = np.round(captured_vertices[:, :2])

        image = np.zeros((100, 100), dtype=np.float32) - 100.
        for v in captured_vertices:
            x, y = int(v[0]), int(v[1])
            image[x, y] = max(image[x, y], v[2])
        image = np.clip(image, 0, 0.1)
        plt.imshow(image)
        plt.pause(0.01)


main()
