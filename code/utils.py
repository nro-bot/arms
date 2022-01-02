from typing import List
import numpy as np
import pyrealsense2 as rs
import cv2
from exceptions import NoFrameException
import tkinter as tk
from PIL import ImageTk, Image


def realse_frame_to_numpy(frame) -> (np.ndarray, np.ndarray):

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


def coord_transform_affine(transform_mtx, pts):

    if len(pts.shape) == 1:
        pts = pts[None, :]

    homog_pts = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    new_homog_pts = np.dot(transform_mtx, homog_pts.T).T
    return new_homog_pts


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


def itos(v):

    lsb = v & 0xFF
    msb = v >> 8
    return lsb, msb


class Colors:
    YES = '#51b442'
    NO = '#a03939'
    NEUTRAL = '#cacaca'
    BLACK = '#2e2e2e'
    ALARM = '#c30b0b'


class Popup:
    def __init__(self,
                 title='',
                 text='',
                 text_color=Colors.BLACK,
                 button_names=['OK','Cancel'],
                 button_colors=[Colors.YES, Colors.NO],
                ):
        def press_gen(name):
            def press():
                self.ret = name
                self.root.destroy()
            return press

        self.ret = None
        self.root = tk.Tk()
        self.root.winfo_toplevel().title(title)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # text
        text_frame = tk.Frame(self.root, pady=4, padx=6)
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid(row=0, sticky='NSEW')
        text_label = tk.Label(text_frame, text=text, fg=text_color,
                               font=('Helvetica','12','normal'))
        text_label.grid(sticky='NSEW')

        # buttons
        buttons_frame = tk.Frame(self.root)
        buttons_frame.grid(row=2, sticky='NSEW', pady=8, padx=8)
        for i, name in enumerate(button_names):
            buttons_frame.grid_columnconfigure(i, weight=1)
            button = tk.Button(buttons_frame, text=name, bg=button_colors[i],
                               font=('Helvetica','12','bold'),
                               width=8, command=press_gen(name))
            button.grid(row=0, column=i, sticky='NS', padx=10)

    def response(self):
        return self()

    def __call__(self):
        self.root.mainloop()
        return self.ret


class ImagePopup(Popup):
    def __init__(self,
                 title='',
                 text='',
                 text_color=Colors.BLACK,
                 button_names=['OK','Cancel'],
                 button_colors=[Colors.YES, Colors.NO],
                 images=[],
                 image_shape=(200,200),
               ):
        super().__init__(title, text, text_color, button_names, button_colors)

        images_frame = tk.Frame(self.root)
        images_frame.grid(row=1, sticky='NESW', padx=6)
        H,W = image_shape

        for i, img_ in enumerate(images):
            if isinstance(img_, str):
                img = Image.open(img_)
            elif isinstance(img_, np.ndarray):
                # image coming from cv so it will be in BGR
                img = Image.fromarray(img_[:,:,::-1])
            else:
                raise TypeError
            img = img.resize((W,H), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            images_frame.grid_columnconfigure(i, weight=1)
            img_canvas = tk.Canvas(images_frame, width=W, height=H, bg='#ffffff')

            img_canvas.image = img
            img_canvas.grid(row=0, column=i, sticky='NS', padx=2)
            img_canvas.create_image(1,1, anchor='nw', image=img)


class VideoPopup(Popup):
    def __init__(self,
                 video_cap,
                 title='',
                 text='',
                 text_color=Colors.BLACK,
                 button_names=['OK','Cancel'],
                 button_colors=[Colors.YES, Colors.NO],
                 image_shape=(375, 500),
               ):
        super().__init__(title, text, text_color, button_names, button_colors)
        self.cap = video_cap
        self.image_shape = image_shape

        video_frame = tk.Frame(self.root)
        video_frame.grid(row=1, sticky='NESW', padx=6)
        H,W = image_shape

        # image coming from cv so it will be in BGR
        img = Image.fromarray(self.cap.read()[:,:,::-1])
        img = img.resize((W,H), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        video_frame.grid_columnconfigure(0, weight=1)
        self.img_canvas = tk.Canvas(video_frame, width=W, height=H, bg='#ffffff')

        self.img_canvas.image = img
        self.img_canvas.grid(row=0, column=0, sticky='NS', padx=2)
        self.img_on_canvas = self.img_canvas.create_image(1,1, anchor='nw', image=img)

        self.root.after(100, self.update_image)

    def update_image(self):
        H,W = self.image_shape
        img = Image.fromarray(self.cap.read()[:,:,::-1])
        img = img.resize((W,H), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.img_canvas.itemconfigure(self.img_on_canvas, image=img)
        self.img_canvas.image = img
        self.root.after(100, self.update_image)


def sample_points_in_workspace(workspace, num_rows_cols=5) -> np.ndarray:

    rows = np.linspace(workspace[0, 0], workspace[0, 1], num_rows_cols)
    cols = np.linspace(workspace[1, 0], workspace[1, 1], num_rows_cols)

    return np.array([[xx, yy] for xx in rows for yy in cols], dtype=np.float32)


def draw_workspace(img, cam_points):

    cv2.line(img, (int(cam_points[0][0]), int(cam_points[0][1])), (int(cam_points[1][0]), int(cam_points[1][1])),
             (255, 0, 0), 5)
    cv2.line(img, (int(cam_points[0][0]), int(cam_points[0][1])), (int(cam_points[2][0]), int(cam_points[2][1])),
             (255, 0, 0), 5)
    cv2.line(img, (int(cam_points[1][0]), int(cam_points[1][1])), (int(cam_points[3][0]), int(cam_points[3][1])),
             (255, 0, 0), 5)
    cv2.line(img, (int(cam_points[2][0]), int(cam_points[2][1])), (int(cam_points[3][0]), int(cam_points[3][1])),
             (255, 0, 0), 5)

    return img


def robot2world_plane(robot_points, robot2world):

    world_points = coord_transform_affine(robot2world, robot_points)
    world_points = np.concatenate([world_points, np.zeros((len(world_points), 1), dtype=np.float32)], axis=1)
    return world_points


def world2pixels(world_points, world2cam, cam_matrix):

    cam_points = coord_transform(world2cam, world_points)
    cam_points = np.dot(cam_matrix, cam_points.T).T
    cam_points[:, :2] /= cam_points[:, 2: 3]
    return cam_points


def robot2world2cam_plane(robot_points, robot2world, world2cam, cam_matrix):

    world_points = robot2world_plane(robot_points, robot2world)
    cam_points = world2pixels(world_points, world2cam, cam_matrix)
    return cam_points


def pixel_to_cam_unknown_z(u, v, focal_length, principal_points):
    # arbitrarily place an object 1 m away from camera
    # we will draw a ray from (0, 0, 0) through this point
    # then we can intersect it with the ground to get a ground point corresponding to pixel (u, v)
    z = 1.
    x = z / focal_length[0] * (u - principal_points[0])
    y = z / focal_length[1] * (v - principal_points[1])
    return np.array([x, y, z], dtype=np.float32)


def calculate_intersection_with_ground(l, cam2world):

    # unit vector corresponding to the direction of our ray
    l /= np.sqrt(np.sum(np.square(l)))

    # get the camera position (0, 0, 0) and a unit vector representing a direction
    # of the ray that hit the pixel of the canvas we selected
    l0 = np.array([0., 0., 0.], dtype=np.float32)

    # calculate the intersection of the ray (l0 + t * l) and the checkerboard
    # let's set the origin of the checkerboard to (0, 0, 0) in world coordinates
    # and its normal to (0, 0, 1)
    l0 = coord_transform(cam2world, l0)[0]
    l = coord_rotate(cam2world, l)[0]
    p0 = np.array([0, 0, 0], dtype=np.float32)
    n = np.array([0, 0, 1], dtype=np.float32)

    # check for rays that never hit the ground
    denom = np.dot(n, l)
    if np.abs(denom) < 1e-6:
        return None

    t = np.dot((p0 - l0), n) / denom
    # point on the intersection of our ray and the ground
    # this is in world space
    p = l0 + l * t
    return p
