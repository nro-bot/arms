import numpy as np
from .xarm_controller import XArmController


class RobotArm:

    def __init__(self, serial_number):

        self.controller = XArmController(serial_number)

