import time

# python -m nuro_arm.examples.record_movements

from nuro_arm.robot.robot_arm import RobotArm
import numpy as np
'''_
raise_up

lower_down

close_grip

open_grip

cube_positions = {
    'A':
    'B':
    'C':
}
#robot.move_hand_to(cube_pos)
'''

def test_move_gripper():
    print('-' * 80)
    print('Test nuro arm library imports correctly and gripper movements work')
    print('-' * 80)

    robot = RobotArm()
    robot.open_gripper()
    time.sleep(0.2)
    robot.close_gripper()

def test_move_xyz():
    print('-' * 80)
    print('Sanity check for moving robot in x,y,z coorindates instead of joint pos')
    print('-' * 80)

    robot = RobotArm()
    print('Starting pose: ', robot.get_hand_pose())

    start_pose_xyz = np.round(robot.get_hand_pose()[0], decimals=4)
    print('-' * 80)
    print('Starting pose: ', start_pose_xyz)

    print('-' * 80)
    print('Now, physically move robot to new pose for recording xyz coords.') 
    print('Sleeping 2 sec while you do so.')
    print('-' * 80)
    robot.passive_mode()
    time.sleep(2)
    
    test_pose_xyz = np.round(robot.get_hand_pose()[0], decimals=4)
    print('Recorded pose: ', test_pose_xyz)
    print('-' * 80)
    time.sleep(1)
    #test_pos = [0.2373, 0.0199, 0.1178]
    #test_pos = [0.159 , 0.0134, 0.2757]
    #robot.move_hand_to(test_pose_xyz)
    #print('Measured pose after movement: ', robot.get_hand_pose())
    for i in range(3):
        print('Moving back to start pose')
        robot.move_hand_to(start_pose_xyz)
        time.sleep(1)
        print('Moving to recorded pose')
        robot.move_hand_to(test_pose_xyz)
        time.sleep(1)

    print('Moving back to start pose')
    robot.move_hand_to(start_pose_xyz)
    time.sleep(1)

    # return to start position
    #robot.move_arm_jpos(START_ARM_JPOS)

    # drop block
    #robot.open_gripper()

#test_move_xyz()

#POS_LEFT = [0.2806 0.0236 0.0425]
#POS_MIDDLE = [0.2806 0.0236 0.0425]
#POS_RIGHT = [ 0.2674 -0.0112  0.0291]
Z_SUPER_HI = 0.10
Z_HI = 0.08
Z_LO = 0.02

OPEN = 0.5
CLOSE = 0.2


# POS from 28 Dec 2021
POS_RIGHT = [0.200, -0.080]
POS_MID = [0.200, -0.010]
POS_LEFT = [ 0.200, 0.070 ]
POS_NEUTRAL = [0.1318, 0.005, 0.1799]

poses_xy = {'left':POS_LEFT,
         'middle':POS_MID,
         'right':POS_RIGHT}


def test_cubes_sequence(cmd):
    print('-' * 80)
    print('Test three positions')
    print('-' * 80)

    robot = RobotArm()
    xyz = np.append(POS_MID, Z_SUPER_HI)
    robot.move_hand_to(xyz)
    robot.set_gripper_state(OPEN)
    pose_xy = poses_xy[cmd]
    xyz = np.append(pose_xy, Z_HI)
    robot.move_hand_to(xyz)
    xyz = np.append(pose_xy, Z_LO)
    robot.move_hand_to(xyz)
    robot.set_gripper_state(CLOSE)
    xyz = np.append(pose_xy, Z_HI)
    robot.move_hand_to(xyz)
    robot.move_hand_to(POS_NEUTRAL)
    robot.set_gripper_state(OPEN)


test_cubes_sequence('left')
test_cubes_sequence('middle')
test_cubes_sequence('right')



















'''
LOL
*cries*
--->  [0.0948 0.0044 0.2419]  <---
--->  [0.0938 0.0043 0.2429]  <---
--->  [0.0945 0.0044 0.2425]  <---
--->  [0.0959 0.0044 0.2417]  <---
--->  [0.0955 0.0044 0.2415]  <---
--->  [0.0952 0.0044 0.2421]  <---

'''
#POS_LEFT = [0.216,  0.0236] #0.0273]
#POS_LEFT = [0.2122, 0.0611]
#POS_LEFT = [0.2112, 0.0599]
#POS_MID = [0.2169, 0.0018] #0.0279]
#POS_RIGHT = [ 0.2133, -0.0398]
#POS_RIGHT = [0.2153, -0.0263]  #0.0279]
