import pyrealsense2 as rs
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import exceptions


class Camera:
	# we have intel realsense D435
	HEIGHT = 640
	WIDTH = 480
	FPS = 30

	def __init__(self):

		pass

	def calc_location(self):

		gh, gw = (9, 7)
		gsize = 0.021

		depth_image, color_image = utils.realse_frame_to_numpy(self.get_frame())
		gray_image = utils.convert_gray(color_image)
		ret, corners = cv2.findChessboardCorners(gray_image, (gh, gw), None)

		if not ret:
			raise exceptions.CheckerboardNotFound()

		objp = np.zeros((gh * gw, 3), np.float32)
		objp[:, :2] = gsize * np.dstack(np.mgrid[1: gw + 1, -1: gh - 1]).reshape(-1, 2)

		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

		ret, rvec, tvec = cv2.solvePnP(objp, corners2, self.color_camera_matrix, self.color_distortion_coeffs)

		world2cam = utils.transformation_matrix(rvec, tvec)
		cam2world = utils.inverse_transformation_matrix(rvec, tvec)

		return rvec, tvec, world2cam, cam2world

	def start(self):

		self.pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.depth, self.HEIGHT, self.WIDTH, rs.format.z16, self.FPS)
		config.enable_stream(rs.stream.color, self.HEIGHT, self.WIDTH, rs.format.bgr8, self.FPS)
		self.profile = self.pipeline.start(config)
		color_profile = self.profile.get_stream(rs.stream.color)
		self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
		self.color_distortion_coeffs = np.array(self.color_intrinsics.coeffs, dtype=np.float32)

		fx = self.color_intrinsics.fx
		fy = self.color_intrinsics.fy
		ppx = self.color_intrinsics.ppx
		ppy = self.color_intrinsics.ppy

		self.color_camera_matrix = np.array([
			[fx, 0, ppx],
			[0, fy, ppy],
			[0, 0, 1]
		], dtype=np.float32)

	def get_frame(self):

		return self.pipeline.wait_for_frames()

	def get_depth_scale(self):

		return self.profile\
			.get_device()\
			.first_depth_sensor()\
			.get_depth_scale()

	def stop(self):

		self.pipeline.stop()
