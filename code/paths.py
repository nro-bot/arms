import os


DEFAULT_CALIBRATION_PATH = "data/calibration.pickle"
DEFAULT_XARM_CONFIG_PATH = "data/xarm_config.npy"
DEFAULT_WORKSPACE_CALIBRATION_PATH = "data/workspace_calibration.pickle"
SCAN_3D_CALIBRATION_TEMPLATE = "data/3d_scan_c{:d}.pickle"

dir_name = os.path.dirname(__file__)
URDF_DIR = os.path.join(dir_name, "assets/urdf")
