import pyrealsense2 as rs
import numpy as np
import cv2
from camera import Camera
import utils

c = Camera()
c.start()


try:
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
