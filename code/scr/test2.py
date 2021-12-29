import time
import numpy as np
from robot.robot_arm import RobotArm


r = RobotArm()
r.home()
r.open_gripper()
r.close_gripper()

r.move_hand_to([0.2, 0.0, 0.05])
r.move_hand_to([0.2, 0.0, 0.015])

r.move_hand_to([0.15, 0.1, 0.05])
r.move_hand_to([0.15, 0.1, 0.015])

r.move_hand_to([0.15, -0.1, 0.05])
r.move_hand_to([0.15, -0.1, 0.015])

r.move_hand_to([0.3, 0.1, 0.05])
r.move_hand_to([0.3, 0.1, 0.015])

r.move_hand_to([0.3, -0.1, 0.05])
r.move_hand_to([0.3, -0.1, 0.015])

#
# r.move_hand_to([0.2, 0.0, 0.05])
# r.open_gripper()
# r.move_hand_to([0.2, 0.0, 0.01])
# r.close_gripper()
# r.home()
# r.open_gripper()

# bottom-left corner = 0.15, 0.1, 0.
# top-left corner = 0.3, 0.11, 0.
# top-right corner = 0.3, -0.09, 0.
# bottom-right corner = 0.15, -0.07, 0.



# r.passive_mode()
# while True:
#     print(r.get_hand_pose())
#     time.sleep(1)
