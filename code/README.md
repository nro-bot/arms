# Setup

Tested with Python 3.6 on Ubuntu 18.

Run this or opencv-python install will take forever.

```
pip install --upgrade pip setuptools wheel
```

Install dependencies.

```
pip install -r requirements.txt
```

I have to run 
```
sudo service fwupd stop
``` 
every time I want to connect to the robot.

# Usage

## Main workflow

You should run scripts from `arms/code`.

Calibrate robot arm. This tells us in which direction the robot arm should be picking.
```
python -m scr.calibrate_arm
```

Calibrate camera. Place a checkerboard so that it is aligned with the robot (i.e. no rotation with respect to the robot base).
This will give us the world coordinate space.
```
python -m scr.calibrate_camera
```

Calibrate robot/workspace. You can remove the checkerboard. Click into the displayed image to select where the robot is pointing.
Try clicking at the point where the tip of the robot's gripper is touching the ground.
```
python -m scr.calibrate_workspace
```

Show workspace.
```
python -m scr.viz.draw_workspace
```

Click into image to make the robot pick and place things.
```
python -m scr.viz.click_move
```

## Other

Run tutorials: `python -m scr.tutorial.1` etc.

Calibrate camera using a checkerboard: `python -m scr.calibrate_camera`, saves data in `data/calibration.pickle`.

Calibrate arm (has graphical UI): `python -m scr.calibrate_arm`, saved data in `data/xarm_config.npy`.

Calculate transform from pixels to robot workspace: `python -m scr.calibrate_workspace`, saved data in `data/workspace_calibration.pickle`.

Draw robot workspace in pixel space: `python -m scr.viz.draw_workspace`.

# Other

* Link to pyrealsense examples: [click here](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples).

* Link to David Klee's nuro-arm repo that we are building off of: [click here](https://github.com/dmklee/nuro-arm).
