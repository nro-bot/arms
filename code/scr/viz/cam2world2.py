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

        cam2world = c.cam2world

        def click_cam2world(event, x, y, flags, param):

            if event != cv2.EVENT_LBUTTONDOWN:
                return

            print("Canvas space (x, y): ({:.4f}, {:.4f}) px.".format(y, x))

            # let's assume that we don't have a depth camera => we don't know how fare away from
            # camera the selected point is
            # simply set z to 1
            z = 1.
            x = z / c.focal_length[0] * (x - c.principal_points[0])
            y = z / c.focal_length[1] * (y - c.principal_points[1])

            # get the camera position (0, 0, 0) and a unit vector representing a direction
            # of the ray that hit the pixel of the canvas we selected
            l0 = np.array([0., 0., 0.], dtype=np.float32)
            l = np.array([x, y, z], dtype=np.float32)
            l /= np.sqrt(np.sum(np.square(l)))

            # calculate the intersection of the ray (l0 + t * l) and the checkerboard
            # let's set the origin of the checkerboard to (0, 0, 0) in world coordinates
            # and its normal to (0, 0, 1)
            l0 = utils.coord_transform(cam2world, l0)[0]
            l = utils.coord_rotate(cam2world, l)[0]
            p0 = np.array([0, 0, 0], dtype=np.float32)
            n = np.array([0, 0, 1], dtype=np.float32)

            # check for rays that never hit the ground
            denom = np.dot(n, l)
            if np.abs(denom) < 1e-6:
                print("Ray does not intersection checkerboard.")
                return

            t = np.dot((p0 - l0), n) / denom
            p = l0 + l * t

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
