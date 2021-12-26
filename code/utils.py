import numpy as np
import pyrealsense2 as rs
import cv2
from exceptions import NoFrameException


def realse_frame_to_numpy(frame):

    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()
    if not depth_frame or not color_frame:
        raise NoFrameException()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return depth_image, color_image


def setup_color_align():

    align_to = rs.stream.color
    return rs.align(align_to)


def update_opencv_window(image):

    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("RealSense", image)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q") or key == 27:
        cv2.destroyAllWindows()
        return True

    return False


def depth_image_to_colormap(image):

    return cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=0.03), cv2.COLORMAP_JET)


def convert_gray(img):

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def transformation_matrix(rvec, tvec):

    mat = np.zeros((4, 4), dtype=np.float32)
    mat[: 3, 3] = tvec.flatten()
    mat[: 3, : 3] = cv2.Rodrigues(rvec)[0]
    mat[3, 3] = 1.
    return mat


def inverse_transformation_matrix(rvec, tvec):

    mat = np.zeros((4, 4), dtype=np.float32)
    rot_mat = cv2.Rodrigues(rvec)[0]
    inv_rot_mat = rot_mat.T
    mat[: 3, 3] = - np.dot(inv_rot_mat, tvec[:, 0])
    mat[: 3, : 3] = inv_rot_mat
    mat[3, 3] = 1.
    return mat


def coord_transform(transform_mtx, pts):

    if len(pts.shape) == 1:
        pts = pts[None, :]

    homog_pts = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    new_homog_pts = np.dot(transform_mtx, homog_pts.T).T
    new_pts = np.true_divide(new_homog_pts[:, :-1], new_homog_pts[:, [-1]])
    return new_pts


def coord_rotate(transform_mtx, pts):

    if len(pts.shape) == 1:
        pts = pts[None, :]

    homog_pts = np.concatenate([pts, np.zeros((len(pts), 1))], axis=1)
    new_homog_pts = np.dot(transform_mtx, homog_pts.T).T
    return new_homog_pts[:, :-1]


def draw_square(img: np.ndarray, x: int, y: int, square_size=20, copy=False):

    size = square_size // 2
    x_limits = [x - size, x + size]
    y_limits = [y - size, y + size]
    for i in range(len(x_limits)):
        x_limits[i] = min(img.shape[0], max(0, x_limits[i]))
    for i in range(len(y_limits)):
        y_limits[i] = min(img.shape[1], max(0, y_limits[i]))

    if copy:
        img = np.array(img, dtype=img.dtype)

    if img.dtype == np.uint8:
        img[x_limits[0]: x_limits[1], y_limits[0]: y_limits[1]] = 255
    else:
        img[x_limits[0]: x_limits[1], y_limits[0]: y_limits[1]] = 1.

    return img


def cv_coords_to_np(x, y):

    return y, x
