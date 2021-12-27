NOTE: model folder is for speech recog 

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

