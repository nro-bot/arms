from typing import Union, List, Tuple, Optional
import os
import pickle
import time
import pyrealsense2 as rs
import open3d as o3d
import cv2
from cv2 import aruco
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

		self.rvec = None
		self.tvec = None
		self.world2cam = None
		self.cam2world = None

		# optional functionality
		self.aruco_dict = None
		self.aruco_params = None
		self.align = None
		self.pc = None

	def calibrate(self, show=False):

		gh, gw = (9, 7)
		gsize = 0.021

		depth_image, color_image = utils.realse_frame_to_numpy(self.get_frame())
		gray_image = utils.convert_gray(color_image)
		ret, corners = cv2.findChessboardCorners(gray_image, (gh, gw), None)

		if show:
			self.show_corners_(gray_image, corners)

		if not ret:
			raise exceptions.CheckerboardNotFound()

		objp = np.zeros((gh * gw, 3), np.float32)
		objp[:, :2] = gsize * np.dstack(np.mgrid[1: gw + 1, -1: gh - 1]).reshape(-1, 2)

		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		corners2 = cv2.cornerSubPix(gray_image, np.array(corners), (11, 11), (-1, -1), criteria)

		if show:
			self.show_corners_(gray_image, corners2)

		ret, rvec, tvec = cv2.solvePnP(objp, corners2, self.color_camera_matrix, self.color_distortion_coeffs)

		world2cam = utils.transformation_matrix(rvec, tvec)
		cam2world = utils.inverse_transformation_matrix(rvec, tvec)

		self.rvec = rvec
		self.tvec = tvec
		self.world2cam = world2cam
		self.cam2world = cam2world

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
		self.focal_length = np.array([fx, fy], dtype=np.float32)
		self.principal_points = np.array([ppx, ppy], dtype=np.float32)

	def get_frame(self):

		return self.pipeline.wait_for_frames()

	def get_aligned_frame(self):
		# the color and depth sensors will see different parts of the world
		# this method will align the depth image with the color image
		# the color image should remain unchanged
		# call setup alignment first
		frame = self.pipeline.wait_for_frames()
		return self.align.process(frame)

	def get_pointcloud(self):
		# get pointcloud in the color camera frame
		depth_frame, color_frame = utils.realse_frame(self.get_frame())
		self.pc.map_to(color_frame)
		points = self.pc.calculate(depth_frame)

		# turn a structured array into a normal array
		tmp = np.asanyarray(points.get_vertices())
		return np.asanyarray(tmp).view(np.float32).reshape(-1, 3)  # xyz

	def get_pointcloud_and_texture(self, frame=None):
		if frame is None:
			frame = self.get_frame()
		# get pointcloud in the color camera frame
		depth_frame, color_frame = utils.realse_frame(frame)
		# map to has to be called before, not after calculating the points!!!
		# the tutorial was wrong
		self.pc.map_to(color_frame)
		points = self.pc.calculate(depth_frame)

		# turn a structured array into a normal array
		v, t = points.get_vertices(), points.get_texture_coordinates()
		verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
		texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
		return verts, texcoords, np.asanyarray(color_frame.get_data())

	def get_pointcloud_open3d(self, return_raw=False):

		cam_vertices, cam_texture, image = self.get_pointcloud_and_texture()

		# not sure if this is because of realsense or open3d
		image = utils.bgr_to_rgb(image)

		# (width, height) => (height, width)
		cam_texture = np.stack([cam_texture[:, 1], cam_texture[:, 0]], axis=-1)
		cam_texture = cam_texture * (image.shape[0], image.shape[1]) + 0.5
		cam_texture = cam_texture.astype(np.uint32)
		u, v = cam_texture[:, 0], cam_texture[:, 1]

		# some depth pixels are outside of the rgb image
		mask = np.logical_and(
			np.logical_and(0 <= u, u <= image.shape[0] - 1),
			np.logical_and(0 <= v, v <= image.shape[1] - 1)
		)
		mask = np.logical_not(mask)
		u = np.clip(u, 0, image.shape[0] - 1)
		v = np.clip(v, 0, image.shape[1] - 1)

		# open3d expects 0 - 1 image, realsense returns 0 - 255
		colors = image[u, v] / 255.
		colors[mask] = 0.

		if return_raw:
			# return data we can pass into open3d
			return cam_vertices, colors
		else:
			# create an open3d pointcloud
			return utils.create_open3d_pointcloud(cam_vertices, colors)

	def get_depth_scale(self):

		return self.profile\
			.get_device()\
			.first_depth_sensor()\
			.get_depth_scale()

	def stop(self):

		self.pipeline.stop()

	def set_calibration(self, tvec, rvec, world2cam, cam2world):

		self.tvec = tvec
		self.rvec = rvec
		self.world2cam = world2cam
		self.cam2world = cam2world

	def save_calibration(self, save_path):

		if self.tvec is None or self.rvec is None or self.cam2world is None or self.world2cam is None:
			raise exceptions.NotCalibratedException()

		d = {
			"tvec": self.tvec,
			"rvec": self.rvec,
			"world2cam": self.world2cam,
			"cam2world": self.cam2world
		}

		save_dir = os.path.dirname(save_path)

		if os.path.isfile(save_dir):
			raise FileExistsError("Save directory is a file.")

		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)

		with open(save_path, "wb") as f:
			pickle.dump(d, f)

	def load_calibration(self, load_path):

		if not os.path.isfile(load_path):
			raise FileNotFoundError("Pickle with calibration data not found.")

		with open(load_path, "rb") as f:
			d = pickle.load(f)

		self.tvec = d["tvec"]
		self.rvec = d["rvec"]
		self.world2cam = d["world2cam"]
		self.cam2world = d["cam2world"]

	def setup_aruco(self):
		# can add more options later
		self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
		self.aruco_params = aruco.DetectorParameters_create()

	def detect_aruco_markers(self, color_frame: np.ndarray) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:

		if self.aruco_dict is None or self.aruco_params is None:
			raise ValueError("Call setup_aruco before detecting markers.")

		gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)  # RGB image in numpy/pyplot is a BGR image in OpenCV
		corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

		if ids is None:
			return None, None

		corners = corners[0]
		ids = ids[0]

		# sort ids, convert into numpy array for consistency
		order = np.argsort(ids)
		ids = np.array(ids, dtype=np.int32)[order]
		corners = np.array(corners, dtype=np.int32)[order]

		return ids, corners

	def setup_alignment(self):
		# call this before calling get_aligned_frame
		self.align = utils.setup_color_align()

	def show_corners_(self, gray_image, corners):

		img = np.array(gray_image)
		for corner in corners:
			x, y = utils.cv_coords_to_np(int(corner[0, 0]), int(corner[0, 1]))
			utils.draw_square(img, x, y, square_size=6)
		plt.imshow(img)
		plt.show()

	def setup_pointcloud(self):

		self.pc = rs.pointcloud()
		# the first frame is always weird
		self.get_pointcloud_and_texture()
		time.sleep(1)
