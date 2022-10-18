import argparse
import cv2
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

        def click_cam2world(event, x, y, flags, param):

            if event != cv2.EVENT_LBUTTONDOWN:
                return

            print("Canvas space (x, y): ({:.4f}, {:.4f}) px.".format(y, x))

            l = utils.pixel_to_cam_unknown_z(x, y, c.focal_length, c.principal_points)
            p = utils.calculate_intersection_with_ground(l, c.cam2world)

            if p is None:
                # no intersection with ground
                print("Ray does not intersect ground.")
                return

            # p is the intersection of the ray and the checkerboard
            print("World space (x, y, z): ({:.4f}, {:.4f}, {:.4f}) m.".format(p[0], p[1], p[2]))

        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("RealSense", click_cam2world)

        while True:

            # Wait for a coherent pair of frames: depth and color
            depth_image, color_image = utils.realse_frame_to_numpy(c.get_frame())
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
    "at the top-right corner of the checkerboard. This version doesn't use the depth camera. "
    "Instead it picks a point on the surface of the checkerboard."
)
parser.add_argument("--load-path", default=paths.DEFAULT_CALIBRATION_PATH, help="Load path for calibration data.")
main(parser.parse_args())
