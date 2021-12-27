'''
Date: 27 Dec 2021
Author: nouyang

Example for reading (multiple) aruco tag.  
The code outputs pose estimate to the shell (or to the file).

Motivation: The openCV documentation is oriented at C++, so it may be difficult to
translate to Python. This is a simple example that reads multiple tags, converts
rotation to Euler, and prints to the commandline. I also explain how to obtain camera
matrix, what an Arucotag library is, how to print a tag, how to set optional
detection PARAMETERS. I also add function to write to CSV file (with semicolon
separator) if you'd like.

NOTE: Be sure to define tag size


Install requirements:
    $ pip install opencv-contrib-python

Example usage:
    $ python minimal_arucotag.py 2
Sample output:
    x: 6.085     y: -17.797   z: 86.09     Rx: 1.778     Ry: -11.036   Rz: -96.712
    x: 6.802     y: -18.344   z: 88.467    Rx: 5.607     Ry: 30.541    Rz: -96.955


NOTE: The `2` stands for "second camera". Usually the first camera will be
your built-in laptop camera, and to select the webcam you put in the `2`
(or try values until the popup opencv window shows the right videostream)
NOTE: Readings are in mm. So, z: 88 = 8.8 cm from camera. x, y origin are
centered at camera also (i think at the center of the lens, not the
sensor).


Ctrl-c, or 'q`, to quit. If writeFlag true, generates file, sample:
    $ cat "2020-04-25 00:54:47_arucotagData.csv"
         Time (sec) - Arucotag x y z Rx Ry Rz; 0.545635; 31.903; -1.28; 118.853; 1.549; -2.216; -0.542;
         Time (sec) - Arucotag x y z Rx Ry Rz; 0.582473; 32.082; -1.374; 116.959; -2.026; 2.759; -0.051; 



With thanks to http://www.philipzucker.com/aruco-in-opencv/
'''

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import sys
from datetime import datetime
import itertools

writeFlag = False # write to CSV file?
strtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
fname = strtime + '_arucotagData.csv'

# help(cv2.aruco)

#---------- 
# Select camera to use (if you have multiple)
if len(sys.argv) > 1:
    CAM_NUM = int(sys.argv[1])
    print('trying camera with id number', CAM_NUM)
    cap = cv2.VideoCapture( CAM_NUM)
else:
    cap = cv2.VideoCapture(0)


#---------- 
# Tag Parameters

# Choose the "library" (tag format) of ArucoTag, e.g. if it is 4x4 square
# For a list of options you can see: 
# --> https://github.com/opencv/opencv_contrib/blob/96ea9a0d8a2dee4ec97ebdec8f79f3c8a24de3b0/modules/aruco/samples/create_board.cpp
# --> It lists: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, ...
# And you can use like so:
# --> ARUCODICT = aruco.Dictionary_get(aruco.DICT_6X6_250)
# If you need to print out a pattern, you can do so with drawMarker()
# Example with tag ID (0) and pixel width (400)
# --> pattern = aruco.drawMarker(ARUCODICT, 0, 400)
# --> cv2.imwrite("aruco_pattern.png", pattern)

ARUCODICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)


PARAMETERS = aruco.DetectorParameters_create()
# Parameters for tag detection; for explanation see:
# --> https://docs.opencv.org/3.4/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
# You can change any parameter from the defaults like so:
# -->  # PARAMETERS.minMarkerPerimeterRate = 0.25


#TAGSIZE = 0.0038 # in meters
TAGSIZE = 0.011 # in meters

NUMTAGS = 1

# Pick a single tag to print (for readability) 
# NOTE: zero-indexed
PRINTINDEX = 0



# --------------------
# Camera parameters (specific to each camera)
# To get them, I found this repository very well documented
# --> https://github.com/smidm/video2calibration#camera-calibration
# --> Basic steps: Print out checkerboard pattern, record video, run calibration.py, get constants
# --> You will get the camera matrix and the distortion coefficients

CAMERAMATRIX = np.array([[521.92676671,   0.,         315.15287785], 
 [  0.,         519.01808261, 193.05763006],
 [  0.,           0.,           1.        ]])


DISTORTIONCOEFFICIENTS =  np.array([ 0.02207713,  0.18641578, -0.01917194, -0.01310851,
                        -0.11910311])



## ------------------------------
# Helpful functions (From learnopencv.com)

# WARNING: Euler angles are human-readable, but the axes often get mixed up.
# If you want to check the axes aren't swapped, 
# To sanity check, I put a tag on a cube of some kind, then set it on the table,
# so that I can constrain the rotation angles I'm checking
# See https://stackoverflow.com/questions/12933284/rodrigues-into-eulerangles-and-vice-versa

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2]) # roll
        y = math.atan2(-R[2,0], sy)  # pitch
        z = math.atan2(R[1,0], R[0,0]) # yaw
    else: # gimbal lock
        print('gimbal lock')
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    rots = np.array([x, y, z])
    rots = np.array([math.degrees(r) for r in rots])

    rots[0] = 180 - rots[0] % 360 
    return rots 


## ------------------------------
# Initialize variables
AXISLABELS = ['x: ', 'y: ', 'z: ', 'Rx: ', 'Ry: ', 'Rz: ']

numFrames = 0
numDetections = 0
outputAngles = np.ones((NUMTAGS, 3))

STARTTIME = datetime.now()

## ------------------------------
while(True):
    numFrames += 1
    rvec, tvec = None, None
    # Capture frame-by-frame
    ret, frame = cap.read(0.)
    # print(frame.shape) #480x640

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    # Detect marker, then return list of ids, and the coordinates of the corners
    # for each marker
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCODICT,
                                                          parameters=PARAMETERS)

    gray = aruco.drawDetectedMarkers(gray, corners)

    # Display the resulting frame
    cv2.imshow('frame', gray)

    if (ids is not None) and (len(ids) == NUMTAGS):
        # The detectMarkers() function returns data in arbitrary order.
        # The following code sorts the data to be ordered by increasing ID
        tagData = zip(ids, corners) # [(id, corners), (id, corners), ...]
        # Since I'm using multiple tags, to keep the output data consistently
        # ordered, I sort by increasing id
        tagData = sorted(tagData, key=lambda x: x[0]) 
        # Now we unzip. The more elegant way is to use zip(*tagData) but for
        # readability I use the below.
        ids = [tag[0] for tag in tagData]
        corners = [tag[1] for tag in tagData]

        # Use built-in algorithm to back out 6D pose 
        # Rotation vector, translation vector
        rvec, tvec, cornerCoeffs  = \
            cv2.aruco.estimatePoseSingleMarkers(corners, TAGSIZE,
                                                CAMERAMATRIX,
                                                DISTORTIONCOEFFICIENTS);

    # Continue if the both tags are detected
    if rvec is not None and rvec.shape[0] == NUMTAGS:
        numDetections += 1
        for tagID in range(NUMTAGS):
            # Convert rvec to rotation matrix
            rotMat, jacob = cv2.Rodrigues(rvec[tagID].flatten())
            # Convert rotation matrix to Euler angles
            rotEuler = rotationMatrixToEulerAngles(rotMat)
            outputAngles[tagID] = rotEuler

        outputAngles = outputAngles.reshape((NUMTAGS, 1, 3))

        # Convert to millimeters
        output = np.concatenate((tvec[PRINTINDEX]*1000, outputAngles[PRINTINDEX])
                                ).flatten()
        output = np.round(output, decimals=3)

        # Add spaces out to 10 chars, so readings line up in nice columns 
        a = ['{0: <10}'.format(axis) for axis in output]

        # Append labels (with zip), then format for printing
        flattened = itertools.chain.from_iterable(zip(AXISLABELS, a)) 
        print(''.join(flattened))

        if writeFlag:
            nowTime = datetime.now()
            dataStr = 'Time (sec) - Arucotag x y z Rx Ry Rz' + '; ' + \
                str((nowTime - STARTTIME).total_seconds()) + '; ' + \
                '; '.join([str(t) for t in output]) + '; \n'

            with open(fname,'a') as outf:
                outf.write(dataStr)
                outf.flush()
    else:
        print('None')



    if cv2.waitKey(1) & 0xFF == ord('q'):
        ENDTIME = datetime.now()
        print('Caught q key, exiting')
        break


# When everything done, release the capture
#strtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
elapsed = (ENDTIME - STARTTIME).total_seconds()
freq = numFrames/elapsed
format = '%Y-%m-%d %H:%M:%S'

print('=============== METADATA ================')
print('Start time: ', STARTTIME.strftime(format))
print('End time: ', ENDTIME.strftime(format))
print('Detections: ', numDetections)
print('Frames/sec (Hz): ', freq)
cap.release()
cv2.destroyAllWindows()

