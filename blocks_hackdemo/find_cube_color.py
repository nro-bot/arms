'''
pip install scikit-learn
'''
import cv2
import cv2.aruco as aruco
import numpy as np
import math
import sys
from datetime import datetime
import colorsys

#from skimage.color import rgb2lab, deltaE_cie76

#import matplotlib.colors as mcolors

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

#print('ids', ids, '\n',  corners)
# Variable 'selected_color' can be any of COLORS['GREEN'], COLORS['BLUE'] or COLORS['YELLOW']
# show_selected_images(images, selected_color, 60)

def match_to_closest_color(sampled_color, max_threshold = 40): 
    
# hue to distinguish colors
# yellow is 40ish
# blue is 200ish
# black is 220sih, but 14ish for value
# value to distinguish black from blue

    # https://stackoverflow.com/questions/2612361/convert-rgb-values-to-equivalent-hsv-values-using-python

    # color is in BGR 
    rgb = sampled_color[::-1] # reverse
    rgb_pct = rgb / 255
    #hsv = mcolors.rgb_to_hsv(rgb)
    hsv_pct = colorsys.rgb_to_hsv(*rgb_pct)
    hsv = np.array(hsv_pct) * 255
    print('rgb', rgb, 'hsv', hsv)
    #print('hsv', [np.int(c) for c in hsv])

    #sampled_color = rgb2lab(np.uint8(sampled_color))
    #image = np.reshape(sampled_color, (1,1,3)) 
    #cv2.cvtColor(image, cv.COLOR_BGR2HSV)

    hue = hsv[0]
    val = hsv[2]

    closest_color = None

    if hue < 60 and hue > 20:
        closest_color = 'yellow'
    if  hue > 180:
        if val > 40:
            closest_color = 'blue'
        else:
            closest_color = 'black'

    return closest_color


image = cv2.imread('screencap2_easy.jpg')
image = cv2.imread('screencap2.jpg')

corners, ids, rejectedImgPoints = aruco.detectMarkers(image, ARUCODICT, parameters=PARAMETERS)

img = aruco.drawDetectedMarkers(image, corners)

while(True):
    cv2.imshow('tags', img)


    # NOTE: assumes cubes are always oriented with tag facing camera
    # directly!!!
    # Determine color by sampling close to tag in known 'correct' direction
    # (which won't sample onto page, other cube, etc.)
    idx = np.where(ids == ID_YEL)[0][0] # pick first occurence of id
    #print(corners)
    corner_coords = corners[idx][0] # such a weird data format.
    xs = corner_coords[:,0]
    ys = corner_coords[:,1]

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    rad = 5
    #print('xmax, xmin', x_max, x_min, 'y', y_max, y_min)
    x, y = int((x_max + x_min) / 2), int(y_min - 10)
    #crop_x1, crop_x2 = int(x + rad), int(x - rad)
    #crop_y1, crop_y2 = int(y + rad), int(y - rad)
    # image[rows, columns]
    #cropped = image[crop_y2:crop_y1, crop_x2:crop_x1]
    crop = image[y-rad:y+rad, x-rad:x+rad]
    average = crop.mean(axis=0).mean(axis=0)
    cv2.imshow('crop', crop)
    # this changes state of image for some reason!!
    annot = cv2.rectangle(image, (x-rad, y-rad), (x+rad, y+rad),
                          (255,0,0),
                          2
                          )
    cv2.imshow('annot', annot)
    print(average)

    color = match_to_closest_color(average)
    print('matched color', color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        ENDTIME = datetime.now()
        print('Caught q key, exiting')
        break
