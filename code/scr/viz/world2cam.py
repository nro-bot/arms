import argparse
import numpy as np
from camera import Camera
import utils
import paths


def main(args):

    c = Camera()
    c.start()

    color_image = None
    depth_image = None

    try:
        try:
            c.load_calibration(args.load_path)
        except FileNotFoundError:
            print("Run 'python -m scr.calibrate' first.")

        world2cam = c.world2cam

        while True:

            world_point = input("x y z: ")
            world_point = np.array([float(x) for x in world_point.split(" ")], dtype=np.float32)
            cam_point = utils.coord_transform(world2cam, world_point)
            cam_point = np.dot(c.color_camera_matrix, cam_point.T).T
            cam_point = cam_point[0]
            cam_point[:2] /= cam_point[2]

            # Wait for a coherent pair of frames: depth and color
            depth_image, color_image = utils.realse_frame_to_numpy(c.get_frame())

            x, y = int(cam_point[0]), int(cam_point[1])
            x, y = utils.cv_coords_to_np(x, y)
            if (0 <= x < color_image.shape[0]) and (0 <= y < color_image.shape[1]):
                # Show images
                if utils.update_opencv_window(utils.draw_square(color_image, x, y, square_size=10, copy=True)):
                    break

    finally:

        # Stop streaming
        c.stop()


parser = argparse.ArgumentParser(
    "Project (x, y, z) coordinates in world points onto the camera canvas. "
    "Try typing '0 0 0', just need to type it twice before a picture appears."
)
parser.add_argument("--load-path", default=paths.DEFAULT_CALIBRATION_PATH, help="Load path for calibration data.")
main(parser.parse_args())
