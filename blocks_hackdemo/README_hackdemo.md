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
import easyhid
en = easyhid.Enumeration()
devices = en.find(vid=1155, pid=22352)
print([dev.description() for dev in devices])
```

Should output similar to 
```
devices ['HIDDevice:\n    /dev/hidraw2 | 483:5750 | MyUSB_HID | LOBOT | 496D626C3331\n    release_number: 513\n    usage_page: 26740\n    usage: 8293\n    interface_number: 0']
```

NOTE: Make sure robot is plugged in to laptop ;P


NOTE: for what it's worth, 
```
$ dmesg | tail
[423316.113725] usb 1-3: new full-speed USB device number 103 using xhci_hcd
[423316.264205] usb 1-3: New USB device found, idVendor=0483, idProduct=5750, bcdDevice= 2.01
[423316.264221] usb 1-3: New USB device strings: Mfr=1, Product=2, SerialNumber=3
[423316.264228] usb 1-3: Product: LOBOT
[423316.264233] usb 1-3: Manufacturer: MyUSB_HID
[423316.264237] usb 1-3: SerialNumber: 496D626C3331
[423316.268102] hid-generic 0003:0483:5750.0071: hiddev0,hidraw2: USB HID v1.10 Device [MyUSB_HID LOBOT] on usb-0000:00:14.0-3/input0
```

And 

```
$ lsusb
Bus 001 Device 103: ID 0483:5750 STMicroelectronics LED badge -- mini LED display -- 11x44
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


## CALIBRATION for IK with nuro arm

        self.joint_names = ('base', 'shoulder','elbow',
                            'wrist','wristRotation', 'gripper'


```
(v3) 14:13:01 nrw@nrw-l390:~/projects/nuro-arm (main *%)$ python -m nuro_arm.robot.calibrate
pybullet build time: Dec  1 2021 18:33:43
hi
en <easyhid.easyhid.Enumeration object at 0x7fe1028e2ee0>
devices ['HIDDevice:\n    /dev/hidraw2 | 483:5750 | MyUSB_HID | LOBOT | 496D626C3331\n    release_number: 513\n    usage_page: 14899\n    usage: 11825\n    interface_number: 0']
Connected to xArm (serial=496D626C3331)
[WARNING] Config file for this xarm could not be found.  Calibration should be performed.
Disconnected xArm

```

```
detected error in: elbow, wrist
```


## SPEECH RECOGNITION

Using Vosk. 
speech recognition quickstart: (runs offline, real time inference, on my cpu laptop easily):
https://alphacephei.com/vosk/install

```
git clone https://github.com/alphacep/vosk-api
cd vosk-api/python/example
wget https://alphacephei.com/kaldi/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 model
pip install sounddevice
python test_microphone.py
```


## SPEECH SYNTHESIS

Using Mozilla TTS.

NOTE: in my bashrc: alias mkenv='python3 -m venv env && startenv && pip3 install --upgrade pip && pip3 install wheel && echo done'

```
mkenv
pip install tts
```

```
$ tts --list_models
$ TTS="tts_models/en/ljspeech/glow-tts"
$ tts --text "What would you like me to do?" --model_name $TTS
 > tts_models/en/ljspeech/glow-tts is already downloaded.
 > Downloading model to /home/nrw/.local/share/tts/vocoder_models--en--ljspeech--multiband-melgan
 > Using model: glow_tts
 > Vocoder Model: multiband_melgan
 > Generator Model: multiband_melgan_generator
 > Discriminator Model: melgan_multiscale_discriminator
 > Text: What would you like me to do?
 > Text splitted to sentences.
['What would you like me to do?']
 > Processing time: 0.7528579235076904
 > Real-time factor: 0.3542274926029484
 > Saving output to tts_output.wav
$ play tts_output.wav
```


###  XYZ MOVEMENTS

checkerboard from close to far: (0.185, 0.297) 1st dim
checkerboard from left to right: (-0.0858, 0.0675) 2nd dim
from bottom to hover over cube: (0.0263 to 0.071) 3rd dim

neutral pose (stable without power): [0.1488 0.0125 0.2536]



### DOCUMENTATION 28 Dec 2021 demo

Video of demo:
https://youtu.be/tg62XLRnbbw

Run:
```
python hackdemo.py
```

Notes:

A few main components.

1. Voice Interface

    Two parts:
    A. Parse speech input
    B. Generate SmartE's voice response

    A. 
        File: test_microphone.py, parse_voice_cmd.py
        is done with vosk library. Runs offline, download a model (already
        included in this repo under voice_model)

        NOTE: MAKE SURE YOU'RE USING LAPTOP MIC AND NOT WEBCAM MIC
        (the latter may be super laggy)
        (change in your computer's sound settings)

    B. 
        File: test_text2speech.py 
        is done with mozilla's TTS library. I pre-decided a few phrases and ran
        them through the TTS library.
        Output stored in speech_output folder, using the words (minus
        punctation) as the filename.

2. Camera: Finding cubes

    A. Which color is the cube? (BLACK, YELLOW, BLUE)
        find_color_xpos() 
        match_to_closest_color()
    B. Which slot is each cube in? (LEFT, MIDDLE, RIGHT)
        find_cube_slot()

    A. 
        I'm using arucotags to figure out where to sample pixels to determine the
        color of a cube. (the ID of the arucotag does not matter, merely that it
        returns the x,y (image space) corners of the tag). I sample a few
        pixels right above each tag.

        The average color of the sampled pixels, I run through a hardcoded HSV to
        determine which of the three colors it is.  Specifically, Use hue to
        separate out the yellow, and then value to separate black from blue. 

   # For all the arucotags, 
    # Find the color (as sampled right above the tag)
    # and fill in dictionary with the color's (avg of the two x's for the corners)
    # which we'll later use to threshold each color into a slot



    B. 
        Then I threshold based on each tag's x,y location to determine which of
        the three "slots" it is in (LEFT, MIDDLE, RIGHT). These are hardcoded x
        thresholds that separate the image into left/middle/right sides.

        Now I know which of three colors is in which of three slots.

3. Robot: Picking up objects 

How to grasp a cube?

A. Grasp primitive
    move_to_grab()

The primitive: Go to neutral pose, open gripper, go to right above cube
spot, lower down, close gripper, lift to right above cube spot, go to
neutral spot, and open gripper.

Implementation:
    After running `python -m nuro_arm.robot.calibrate` we can move the
    robot in x,y,z (robot frame) coordinates thanks to nuro library*

        *which I think relies on knowledge of robot geometry, given joint poses
        (which are read directly from the lxd150 servos), runs through IK built into
        pybullet library, and spits out x,y,z

    I hardcode three (x,y) (robot frame) coordinates, for each of the three
    slots. These slots are widely spaced so the gripper can open between the
    three cubes without hitting ones to either side.

    I hardcode an open and closed position (value from 0 to 1) so that the
    gripper doesn't push any of the cubes around.

    Finally I hardcode a NEUTRAL position where, even if the robot is off, the
    servos will hold position.

    The cubes are arranged in a semicircle (instead of flat line) because I do
    not control orientation of the wrist.

        (NOTE: I have sharpied on the robot which joint is which name in the nuro
        arm library)

    The grasp primitive then takes a position (left / right / middle), from
    which it gets an (x,y) (robot space) position, and executes the  grasp
    primitive.





NOTE: HARDCODED:
There are some hardcoded assumptions.



3A. TO CHANGE HARDCODE:
to find positions. you can run in ipython
```python
 from nuro_arm.robot.robot_arm import RobotArm
 import numpy as np

    robot = RobotArm(); 
    robot.passive_mode() ;   
    # now robot servos do not fight you. physically move robot arm to desired pos
    test_pose_xyz = np.round(robot.get_hand_pose()[0], decimals=4); 
    print('---> ', test_pose_xyz, ' <---')
```

Note that these vals will chagne slightly each time you instantiate the
RobotArm() object. In the end I kind of eyed how far apart I wanted the
slots to be, vs the x,y (robot space) coordinates of the far sides of the
checkerboard.


See also:
```
python -m nuro_arm.robot.record_movements
python -m nuro_arm.examples.move_arm_with_gui
```

2B. HARDCODED X THRESHOLDS for separating out slots

Run inspect_camera_img(), move your cursor around. The x,y of the cursor is
printed on the bottom of the window.w:w


NOTE: Generate text to speech phrases with test_text2speech.py 

NOTE: For tags, print from pdf in this repo, or generate the tags as
described in read_aruco.py. Tag ID does not matter for hackdemo. 
We just use the x,y (image coords) of the tags.
However, if you want x,y,z (camera lens as origin) coords, must know tag
size. See read_aruco.py TAGSIZE variable. The tags show up in inkscape as 15mm, printed
out and measured as 11mm. (The camera matrix, distortion matrix in there
are from some random webcam, works well enough for other webcam if xyz
accuracy not critical)


NOTE: To see what the sampled rgions are like. see find_cube_color.py. Or
annotated_screencap in current directory -- small blue square shows where
we're sampling to determine cube color.


# Nuro Arm library usage notes

Can use simulation mode.

```bash
senv # start virual env with opencv etc. installed

(v3) 14:45:32 nrw@nrw-l390:~/projects/arms/blocks_hackdemo (main *%)$ python -m nuro_arm.examples.move_arm_with_gui --sim
```
