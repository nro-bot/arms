import pyrealsense2 as rs
import numpy as np
import cv2
from camera import Camera
import utils


c = Camera()
c.start()

color_image = None
depth_image = None

try:
    rvec, tvec, world2cam, cam2world = c.calc_location()


    def click_cam2world(event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        y, x = x, y
        window_size = 20
        z = depth_image[x - window_size: x + window_size, y - window_size: y + window_size]
        z = np.mean(z[z != 0])

        vec = np.array([x, y, z, 1.], dtype=np.float32)
        vec2 = np.matmul(vec, cam2world)
        print(x, y, z)
        print(vec2)


    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("RealSense", click_cam2world)

    while True:

        # Wait for a coherent pair of frames: depth and color
        depth_image, color_image = utils.realse_frame_to_numpy(c.get_frame())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = utils.depth_image_to_colormap(depth_image)
        images = np.hstack((color_image, depth_colormap))

        # Show images
        if utils.update_opencv_window(images):
            break

finally:

    # Stop streaming
    c.stop()
