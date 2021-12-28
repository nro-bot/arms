import cv2
import cv2.aruco as aruco
import numpy as np
import math
import sys
from datetime import datetime
import itertools


'''
#---------- 
# Select camera to use (if you have multiple)
if len(sys.argv) > 1:
    CAM_NUM = int(sys.argv[1])
    print('trying camera with id number', CAM_NUM)
    cap = cv2.VideoCapture( CAM_NUM)
else:
    cap = cv2.VideoCapture(0)


# lower bound and upper bound for Green color
lower_bound = np.array([50, 20, 20])
upper_bound = np.array([100, 255, 255])
# find the colors within the boundaries

while(True):
    ret, frame = cap.read(0.)
    # print(frame.shape) #480x640

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    segmented_img = cv2.bitwise_and(frame, frame, mask=mask)

    #define kernel size
kernel = np.ones((7,7),np.uint8)
# Remove unnecessary noise from mask
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
'''

ID_BLK = 0
ID_BLU = 1
ID_YEL = 2
ARUCODICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)


PARAMETERS = aruco.DetectorParameters_create()

image = cv2.imread('screencap2_easy.jpg')

corners, ids, rejectedImgPoints = aruco.detectMarkers(image, ARUCODICT, parameters=PARAMETERS)

#print('ids', ids, '\n',  corners)
img = aruco.drawDetectedMarkers(image, corners)
while(True):
    cv2.imshow('frame', img)


    # NOTE: assumes cubes are always oriented with tag facing camera
    # directly!!!
    # Determine color by sampling close to tag in known 'correct' direction
    # (which won't sample onto page, other cube, etc.)
    idx = np.where(ids == ID_YEL)[0][0] # pick first occurence
    print(corners)
    corner_coords = corners[idx][0] # such a weird data format.
    xs = corner_coords[:,0]
    ys = corner_coords[:,1]

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    rad = 5
    print('xmax, xmin', x_max, x_min, 'y', y_max, y_min)
    x, y = int((x_max + x_min) / 2), int(y_min - 10)
    #crop_x1, crop_x2 = int(x + rad), int(x - rad)
    #crop_y1, crop_y2 = int(y + rad), int(y - rad)
    # image[rows, columns]
    #cropped = image[crop_y2:crop_y1, crop_x2:crop_x1]
    annot = cv2.rectangle(image, (x-rad, y-rad), (x+rad, y+rad),
                          (255,0,0),
                          2
                          )
    #image[y-rad:y+rad, x-rad:x+rad]
    #cv2.imshow('crop', cropped)
    cv2.imshow('crop', annot)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        ENDTIME = datetime.now()
        print('Caught q key, exiting')
        break
