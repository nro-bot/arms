import pyrealsense2 as rs
import numpy as np
import cv2
from camera import Camera
import utils


c = Camera()
c.start()
depth_scale = c.get_depth_scale()

print("Depth Scale is: ", depth_scale)

align = utils.setup_color_align()

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = c.get_aligned_frame()
        depth_image, color_image = utils.realse_frame_to_numpy(frames)

        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        aligned_depth_image, aligned_color_image = utils.realse_frame_to_numpy(aligned_frames)

        depth_colormap = utils.depth_image_to_colormap(depth_image)
        aligned_depth_colormap = utils.depth_image_to_colormap(aligned_depth_image)

        # TODO: make the downsampled depth fit
        images1 = np.hstack((color_image, depth_colormap))
        images2 = np.hstack((aligned_color_image, aligned_depth_colormap))
        images = np.vstack((images1, images2))

        if utils.update_opencv_window(images):
            break

finally:
    c.stop()
