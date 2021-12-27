
## HOW TO MOVE ARM (31 Oct 2021)

Use this code
https://gist.github.com/maximecb/7fd42439e8a28b9a74a4f7db68281071

To set up on ubuntu (20.04) need to do some usb permissions stuff

Create rules file
```
sudo vi /usr/lib/udev/rules.d/99-xarm.rules
SUBSYSTEM=="hidraw", ATTRS{product}=="LOBOT", GROUP="dialout", MODE="0666"
```

Add user to `dialout` group
```
sudo usermod -a -G dialout $USER
sudo udevadm control --reload-rules && sudo udevadm trigger
```

```
pip install easyhid
```

maybe needed? 
```
sudo apt-get install libhidapi-hidraw0 libhidapi-libusb0
````

Troubleshooting: 
For sanity check if having issues, try 
```
print([dev.description for dev in devices])
```

Should output similar to 
```
devices ['HIDDevice:\n    /dev/hidraw2 | 483:5750 | MyUSB_HID | LOBOT | 496D626C3331\n    release_number: 513\n    usage_page: 26740\n    usage: 8293\n    interface_number: 0']
```

## REALSENSE camera (31 Oct 2021)

on the other laptop, for realsense camera, per ondrej:
https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py

"You only need pip install pyrealsense2, no need to install SDK and kernel patch"
also call
```
$ activate_realsense
```


## Other hardware notes

Servo motor specs:
(used in our arm)
https://ozrobotics.com/shop/hiwonder-lx-15d-intelligent-serial-bus-servo-with-rgb-indicator-for-displaying-robot-status/ 

Other cool servomotor: https://robosavvy.com/store/robotis-dynamixel-xl330-m288-t.html (new dynamixels)
