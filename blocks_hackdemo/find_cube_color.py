'''
pip install scikit-learn
'''
import cv2
import cv2.aruco as aruco
import numpy as np
import math
import sys
from datetime import datetime
import itertools

from skimage.color import rgb2lab, deltaE_cie76


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




# Get green color
blue = np.uint8([[[0, 0, 255]]])
# Convert Green color to Green HSV
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
# Print HSV Value for Green color
print(hsv_blue)
h = hsv_blue
lower = [h-10, 100, 100]
upper = [h+10, 255, 255]


ARUCODICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)


PARAMETERS = aruco.DetectorParameters_create()

#print('ids', ids, '\n',  corners)
# Variable 'selected_color' can be any of COLORS['GREEN'], COLORS['BLUE'] or COLORS['YELLOW']
# show_selected_images(images, selected_color, 60)

def match_to_closest_color(sampled_color, max_threshold = 40): 
    
    '''
    COLORS = {
        'BLACK': [0, 0, 0],
        'BLUE': [0, 0, 128],
        'YELLOW': [255, 255, 0]
    }
    '''
    COLORS = {
        'BLACK': [0, 0, 0],
        'BLUE': [45, 77, 134],
        'YELLOW': [150, 130, 80]
    }


    #image_color = get_colors(image, number_of_colors, False)
    sampled_color = rgb2lab(np.uint8(sampled_color))

    closest_color = None

    min_diff = np.inf
    for color, val in COLORS.items():
        curr_color = rgb2lab(np.uint8(val))
        diff = deltaE_cie76(sampled_color, curr_color)
        if diff < max_threshold:
            if diff < min_diff:
                min_diff = diff
                closest_color = color

    return closest_color


# note: for smarter color picker alg, see https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
# https://towardsdatascience.com/color-identification-in-images-machine-learning-application-b26e770c4c71

image = cv2.imread('screencap2_easy.jpg')

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
    '''
    # this changes state of image for some reason!!
    annot = cv2.rectangle(image, (x-rad, y-rad), (x+rad, y+rad),
                          (255,0,0),
                          2
                          )
    cv2.imshow('annot', annot)
    '''
    print(average)

    color = match_to_closest_color(average)
    print('matched color', color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        ENDTIME = datetime.now()
        print('Caught q key, exiting')
        break




