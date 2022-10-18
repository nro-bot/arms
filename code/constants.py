import numpy as np


GRIPPER_CLOSED = 0
GRIPPER_OPENED = 1

WORKSPACE = np.array([
    [0.15, 0.25],
    [-0.1, 0.1]
], dtype=np.float32)
Z_UP = 0.05
Z_DOWN = 0.015
