'''
28 Dec 2021
author: nouyang
'''

import argparse
import os
import queue
import sounddevice as sd
import vosk
import sys
import re
#import time
import cv2
import cv2.aruco as aruco
import numpy as np
import math
from datetime import datetime
import colorsys

CAMERA_NUM = 2

from nuro_arm.robot.robot_arm import RobotArm


def move_to_grab(position, grab=True):
    # position can ONLY be: 
    # 'left', 'middle', or 'right'
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

    robot = RobotArm()
    xyz = np.append(POS_MID, Z_SUPER_HI)
    robot.move_hand_to(xyz)
    robot.set_gripper_state(OPEN)
    cube_xy = poses_xy[position]
    xyz = np.append(cube_xy, Z_HI)
    robot.move_hand_to(xyz)
    if grab:
        xyz = np.append(cube_xy, Z_LO)
        robot.move_hand_to(xyz)
        robot.set_gripper_state(CLOSE)
        xyz = np.append(cube_xy, Z_HI)
        robot.move_hand_to(xyz)
    robot.move_hand_to(POS_NEUTRAL)
    robot.set_gripper_state(OPEN)

X_THRES_1 = 223
X_THRES_2 = 350
def inspect_camera_img():
    # Human Helper fxn for finding thresholds
    cam = cv2.VideoCapture(CAMERA_NUM)
    print('Showing image. Pick your x thresholds for the three slots.')
    while True:
        ret, frame = cam.read(0.)
        if not ret:
            print("failed to grab frame")
        cv2.imshow("camera img", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("ESC key hit, closing...")
            break
        elif k & 0xFF == ord('q') :
            print("q key hit, closing...")
            break
    cam.release()
    cv2.destroyAllWindows()
    # manually redefine these as needed 
    print('-' * 80)
    print(f'The x thresholds you set are: {X_THRES_1}, {X_THRES_2}')
    print('-' * 80)


def find_cube_slot(desired_color, color_x_coords, x_thres_1, x_thres_2):
    slot =  None # 'left', 'middle', or 'right'
    color_slots = {
        'black': None,
        'blue': None,
        'yellow': None
        }

    for color, xcoord in color_x_coords.items():
        if xcoord is not None:
            if xcoord < x_thres_1:
                slot = 'left'
            elif xcoord < x_thres_2:
                slot = 'middle'
            else:
                slot = 'right'
            color_slots[color] = slot

    print('-' * 80)
    print('output of find_cube_slot: ', color_slots)
    print('-' * 80)
    return color_slots[desired_color]


def match_to_closest_color(color_sample): 
    # NOTE: has to be tweaked when it's daylight or nighttime T^T
    # match to yellow, blue, or black

    # yellow is 40ish # blue is 200ish # black is 220sih, 
    # but but 14ish for value
    # value to distinguish black from blue
    rgb = color_sample[::-1] # reverse, color is in BGR from opencv
    rgb_pct = rgb / 255
    hsv_pct = colorsys.rgb_to_hsv(*rgb_pct) # fxn takes 3 values, so unpack list with *
    hsv = np.array(hsv_pct) * 255
    print('rgb', rgb, 'hsv', hsv)

    hue = hsv[0]
    val = hsv[2]

    closest_color = None

    if 10 < hue < 40:
        closest_color = 'yellow'
    if hue >= 50:
        if val > 90:
            closest_color = 'blue'
        else:
            closest_color = 'black'

    return closest_color

def find_color_xcoords():
    # NOTE: assumes cubes are always oriented with tag facing camera
    # directly!!!

    # Find arucotags
    # Determine color by sampling close to tag in known 'correct' direction
    # (which won't sample onto page, other cube, etc.)

    # and fill in dictionary with x coordinate of each color 
    # we'll later use the x coord to determine which slot a color is in

    # x, color for each of three things
    color_x_coords = {
        'black': None,
        'blue': None,
        'yellow': None,
    }

    # NOTE: all cubes have same tag
    # ID_CUBE = 0

    cam = cv2.VideoCapture(CAMERA_NUM)
    ret, image = cam.read(0.)
    print('captured an image')
    cam.release()

    ARUCODICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    PARAMETERS = aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(image, ARUCODICT, parameters=PARAMETERS)
    #img = aruco.drawDetectedMarkers(image, corners)
    #cv2.imshow('tags', img)
    #print(corners)

    print('found ids', ids)

    '''
    list_annots = []
    '''
    DRAW_SAMPLED_REGIONS = False
    for n, id in enumerate(ids):
        tag_corner_coords = corners[n][0] # such a weird data format.
        xs = tag_corner_coords[:,0] 
        ys = tag_corner_coords[:,1]

        rad = 5 # sample radius

        # pick horizotonal center of tag, then go up a bit from top of tag
        x = int(np.average(xs))

        y_min = np.min(ys) # min y = highest point of tag
        y_offset = 8
        y = int(y_min - y_offset)

        # CV2 format: image[rows, columns]
        crop = image[y-rad:y+rad, x-rad:x+rad]
        cv2.imshow('For debugging: part of image sampled for color', crop)

        if DRAW_SAMPLED_REGIONS:
            # NOTE: for debugging only!: draw blue rectangles around where
            # we're sampling 

            # WARNING: this WILL CHANGE the sampled color!  For some reason,
            # despite use of copy(), the blue rectangle will get
            # included in the sampled pixels

            annot = cv2.rectangle(image.copy(), (x-rad, y-rad), (x+rad, y+rad),
                                  (255,0,0),
                                  2
                                  )
            list_annots.append(annot)

        average = crop.mean(axis=0).mean(axis=0)
        print('-' * 40)
        print(f'RGB value of Average for id {id} is {average}')
        print('-' * 40)

        color = match_to_closest_color(average)
        print('matched color', color)

        if color is not None:
            color_x_coords[color] = x
        else:
            print(f'ERROR: for the {n} tag, out of {len(ids)} tags,' \
                f' I found no color.')

    if DRAW_SAMPLED_REGIONS:
        while True:
            for annot_img in list_annots:
                cv2.imshow('annot', annot_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('Caught q key, exiting')
                    break
            if cv2.waitKey(1) & 0xFF == ord('x'):
                print('Caught x key, exiting')
                break

    print('for each color, i found these x coords:', color_x_coords)
    return color_x_coords


def pick_cube_by_color(color):
    color_x_coords = find_color_xcoords()
    print(f'found positions: {color_x_coords}')

    slot = find_cube_slot(color, color_x_coords, X_THRES_1, X_THRES_2)
    print(f'found slot {slot} for color {color}')

    try:
        move_to_grab(slot)
        return True
    except:
        print('failed to grab')
        return False

#STATE_PICK = False

if __name__ == '__main__':
    #print('inspect camera image')
    #x_thres_1, x_thres_2 = inspect_camera_img()

    #find_color_xcoords('yellow')

    #color_x_coords = find_color_xcoords()
    #slot = find_cube_slot('yellow', color_x_coords, X_THRES_1, X_THRES_2)

    #move_to_grab('left', grab=False)

    FLAG_RUN = False
    FLAG_RUN = True
    if FLAG_RUN:
        phrases = ["Hi there!", 
                   "Say that again?",
                   "Okay, I can do that",
                   "Okay, getting the blue cube",
                   "Okay, getting the yellow cube",
                   "Okay, getting the black cube"]


        folder = 'speech_output' 

        reponse_phrase_idxs = {
                'smarty':0,
                'blue':3,
                'yellow':4,
                'black':5
        }

        sound_filenames = []

        for phrase in phrases:
            clean_filename = re.sub(r'\W+', '', phrase)
            filename = f'{folder}/{clean_filename}.wav'
            sound_filenames.append(filename)

        q = queue.Queue()

        def callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))

        def find_keyword(voice_result):
            keywords = ['marty', 'smarty', 'smart']
            if any(keyword in voice_result for keyword in keywords):
                print('-' * 40)
                print('keyword recongized')
                print('this is voice result', voice_result)
                print('-' * 40)

                response_idx = response_phrase_idxs['smarty']
                sound_file = sound_filenames[idx]
                os.system(f'play "{sound_file}"')

                q.queue.clear()
                return True
            return False


        def parse_for_phrases(voice_result):
            print('-' * 40)
            print('this is voice result', voice_result)
            print('-' * 40)

            colors = ['blue', 'yellow', 'black']
            words = voice_result.split(' ')
            found_color = None

            for color in colors:
                if color in words:
                    found_color = color
                    print(f'Parsed {color}!')

                    response_idx = response_phrase_idxs[color]
                    sound_file = sound_filenames[idx]
                    os.system(f'play "{sound_file}"')

                    q.queue.clear()
            return found_color
            

        my_model = "voice_model" # for speech recognition

        if not os.path.exists(my_model):
            print ("Please download a model for your language from https://alphacephei.com/vosk/models")
            print ("and unpack as 'voice_model' in the current folder.")
        model = vosk.Model(my_model)

        device = None
        device_info = sd.query_devices(device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        samplerate = int(device_info['default_samplerate'])

        state_triggered = False

        try:
            with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=device, dtype='int16',
                                    channels=1, callback=callback):
                print('#' * 80)
                print('Press Ctrl+C to stop the recording')
                print('#' * 80)
                print(f'Samplerate: {samplerate}, device: {device}')

                rec = vosk.KaldiRecognizer(model, samplerate)

                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        print('state', state_triggered)
                        sentence = rec.Result()
                        if not state_triggered:
                            found_keyword = find_keyword(sentence)
                            if found_keyword:
                                state_triggered = True
                        elif state_triggered:
                            #completed_cmd, color = parse_for_phrases(sentence)
                            color = parse_for_phrases(sentence)
                            print(f'getting a cube by color {color}')
                            completed_cmd = pick_cube_by_color(color)
                            if completed_cmd:
                                print(f'going back to listening for "smarty"')
                                state_triggered = False #listen for smarty again
                    else:
                        print(rec.PartialResult())


        except KeyboardInterrupt:
            print('\nDone')
        except Exception as e:
            print('Exception: ', e)

