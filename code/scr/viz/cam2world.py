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

        world2cam = c.world2cam
        cam2world = c.cam2world

        def click_cam2world(event, x, y, flags, param):

            if event != cv2.EVENT_LBUTTONDOWN:
                return

            y, x = x, y
            print("Canvas space (x, y): ({:.4f}, {:.4f}) px.".format(y, x))
            window_size = 20
            z = depth_image[x - window_size: x + window_size, y - window_size: y + window_size]
            z = np.mean(z[z != 0]) / 1000.

            # TODO: why do I do this?
            x, y = y, x

            x = z / c.focal_length[0] * (x - c.principal_points[0])
            y = z / c.focal_length[1] * (y - c.principal_points[1])

            vec = np.array([x, y, z])
            vec2 = utils.coord_transform(cam2world, vec)[0]

            print("Camera space (x, y, z): ({:.4f}, {:.4f}, {:.4f}) m.".format(x, y, z))
            print("World space (x, y, z): ({:.4f}, {:.4f}, {:.4f}) m.".format(vec2[0], vec2[1], vec2[2]))

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
    "at the top-right corner of the checkerboard."
)
parser.add_argument("--load-path", default=paths.DEFAULT_CALIBRATION_PATH, help="Load path for calibration data.")
main(parser.parse_args())
