import pyrealsense2 as rs
import numpy as np
import cv2
from camera import Camera
import utils


c = Camera()
c.start()
depth_scale = c.get_depth_scale()

print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align = utils.setup_color_align()


# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = c.get_frame()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        depth_image, color_image = utils.realse_frame_to_numpy(aligned_frames)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = utils.depth_image_to_colormap(depth_image)
        images = np.hstack((bg_removed, depth_colormap))

        if utils.update_opencv_window(images):
            break

finally:
    c.stop()
