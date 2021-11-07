import pyrealsense2 as rs
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Camera:
	# we have intel realsense D435
	HEIGHT = 640
	WIDTH = 480
	FPS = 30

	def __init__(self):

		pass

	def start(self):

		self.pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.depth, self.HEIGHT, self.WIDTH, rs.format.z16, self.FPS)
		config.enable_stream(rs.stream.color, self.HEIGHT, self.WIDTH, rs.format.bgr8, self.FPS)
		self.profile = self.pipeline.start(config)

	def get_frame(self):

		return self.pipeline.wait_for_frames()

	def get_depth_scale(self):

		return self.profile\
			.get_device()\
			.first_depth_sensor()\
			.get_depth_scale()

	def stop(self):

		self.pipeline.stop()
