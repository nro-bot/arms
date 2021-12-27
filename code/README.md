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

# Usage

Run tutorials: `python -m scr.tutorial.1` etc.

Calibrate camera: `python -m scr.calibrate_camera`, saves data in `data/calibration.pickle`.

Calibrate arm (has graphical UI): `python -m scr.calibrate_arm`, saved data in `data/xarm_config.npy`.

# Other

* Link to pyrealsense examples: [click here](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples).

* Link to David Klee's nuro-arm repo that we are building off of: [click here](https://github.com/dmklee/nuro-arm).
