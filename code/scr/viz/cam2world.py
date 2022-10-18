import argparse
import cv2
import numpy as np
from camera import Camera
import utils
import paths


def main(args):

    c = Camera()
    c.start()
    c.setup_alignment()

    color_image = None
    depth_image = None

    try:
        try:
            c.load_calibration(args.load_path)
        except FileNotFoundError:
            print("Run 'python -m scr.calibrate' first.")

        cam2world = c.cam2world

        def click_cam2world(event, x, y, flags, param):

            if event != cv2.EVENT_LBUTTONDOWN:
                return

            print("Canvas space (x, y): ({:.4f}, {:.4f}) px.".format(y, x))

            # use depth to calculate how far away the pixel we selected is from camera
            window_size = 5
            tmp_x, tmp_y = utils.cv_coords_to_np(x, y)
            z = depth_image[tmp_x - window_size: tmp_x + window_size, tmp_y - window_size: tmp_y + window_size]
            z = np.mean(z[z != 0]) / 1000.

            x = z / c.focal_length[0] * (x - c.principal_points[0])
            y = z / c.focal_length[1] * (y - c.principal_points[1])

            # point in the camera space
            point_camera = np.array([x, y, z], dtype=np.float32)
            # point in the world space
            point_world = utils.coord_transform(cam2world, point_camera)[0]

            print("Camera space (x, y, z): ({:.4f}, {:.4f}, {:.4f}) m.".format(
                point_camera[0], point_camera[1], point_camera[2])
            )
            print("World space (x, y, z): ({:.4f}, {:.4f}, {:.4f}) m.".format(
                point_world[0], point_world[1], point_world[2])
            )

        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("RealSense", click_cam2world)

        while True:

            # Wait for a coherent pair of frames: depth and color
            depth_image, color_image = utils.realse_frame_to_numpy(c.get_aligned_frame())
            depth_colormap = utils.depth_image_to_colormap(depth_image)
            images = np.hstack((color_image, depth_colormap))

            # Show images
            if utils.update_opencv_window(images):
                break

    finally:

        # Stop streaming
        c.stop()


parser = argparse.ArgumentParser(
    "Click in to the image. The script will print a world coordinate centered "
    "at the top-right corner of the checkerboard."
)
parser.add_argument("--load-path", default=paths.DEFAULT_CALIBRATION_PATH, help="Load path for calibration data.")
main(parser.parse_args())
