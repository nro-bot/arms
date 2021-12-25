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

finally:

    # Stop streaming
    c.stop()
