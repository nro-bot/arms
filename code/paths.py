import os


DEFAULT_CALIBRATION_PATH = "data/calibration.pickle"
DEFAULT_XARM_CONFIG_PATH = "data/xarm_config.npy"

dir_name = os.path.dirname(__file__)
URDF_DIR = os.path.join(dir_name, "assets/urdf")
print(URDF_DIR)
