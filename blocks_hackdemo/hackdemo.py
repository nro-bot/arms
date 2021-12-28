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

phrases = ["Hi there!", 
           "Say that again?",
           "Okay, I can do that",
           "Okay, getting the blue cube",
           "Okay, getting the yellow cube",
           "Okay, getting the black cube"]


'''
self_phrases = ["okay getting the blue que",
                "okay getting the yellow que",
                "okay getting the black que",
                ]
'''


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
    pose_xy = poses_xy[position]
    xyz = np.append(pose_xy, Z_HI)
    robot.move_hand_to(xyz)
    if grab:
        xyz = np.append(pose_xy, Z_LO)
        robot.move_hand_to(xyz)
        robot.set_gripper_state(CLOSE)
        xyz = np.append(pose_xy, Z_HI)
        robot.move_hand_to(xyz)
    robot.move_hand_to(POS_NEUTRAL)
    robot.set_gripper_state(OPEN)

X_THRES_1 = 223
X_THRES_2 = 350
def inspect_camera_img():
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


def find_cube_slot(desired_color, color_x_positions, x_thres_1, x_thres_2):
    slot =  None # 'left', 'middle', or 'right'
    color_slots = {
        'black': None,
        'blue': None,
        'yellow': None
        }

    for color, pos in color_x_positions.items():
        if pos is not None:
            if pos < x_thres_1:
                slot = 'left'
            elif pos < x_thres_2:
                slot = 'middle'
            else:
                slot = 'right'
            color_slots[color] = slot

    print('-' * 80)
    print('output of find_cube_slot: ', color_slots)
    print('-' * 80)
    return color_slots[desired_color]


def match_to_closest_color(sampled_color, max_threshold = 40): 
    # NOTE: has to be tweaked when it's daylight or nighttime T^T
    # match to yellow, blue, or black

    # yellow is 40ish # blue is 200ish # black is 220sih, 
    # but but 14ish for value
    # value to distinguish black from blue
    rgb = sampled_color[::-1] # reverse, color is in BGR from opencv
    rgb_pct = rgb / 255
    hsv_pct = colorsys.rgb_to_hsv(*rgb_pct)
    hsv = np.array(hsv_pct) * 255
    print('rgb', rgb, 'hsv', hsv)

    hue = hsv[0]
    val = hsv[2]

    closest_color = None

    if hue < 40 and hue > 10:
        closest_color = 'yellow'
    if  hue >= 50:
        if val > 90:
            closest_color = 'blue'
        else:
            closest_color = 'black'

    return closest_color

def find_color_xpos():
    # NOTE: assumes cubes are always oriented with tag facing camera
    # directly!!!
    # Determine color by sampling close to tag in known 'correct' direction
    # (which won't sample onto page, other cube, etc.)

    # x, color for each of three things
    color_x_positions = {
        'black': None,
        'blue': None,
        'yellow': None,
    }

    # NOTE: all cubes have same tag
    ID_CUBE = 0

    cam = cv2.VideoCapture(CAMERA_NUM)
    ret, image = cam.read(0.)
    print('captured an image')
    cam.release()

    ARUCODICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    PARAMETERS = aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(image, ARUCODICT, parameters=PARAMETERS)
    #img = aruco.drawDetectedMarkers(image, corners)
    #cv2.imshow('tags', img)

    print('found ids', ids)
    list_annots = []
    for n, id in enumerate(ids):
        #print(corners)
        corner_coords = corners[n][0] # such a weird data format.
        xs = corner_coords[:,0]
        ys = corner_coords[:,1]

        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)

        rad = 5 # sample rad
        #print('xmax, xmin', x_max, x_min, 'y', y_max, y_min)
        y_offset = 8
        x, y = int((x_max + x_min) / 2), int(y_min - y_offset)
        #crop_x1, crop_x2 = int(x + rad), int(x - rad)
        #crop_y1, crop_y2 = int(y + rad), int(y - rad)
        # image[rows, columns]
        #cropped = image[crop_y2:crop_y1, crop_x2:crop_x1]
        crop = image[y-rad:y+rad, x-rad:x+rad]
        cv2.imshow('crop', crop)


        average = crop.mean(axis=0).mean(axis=0)
        print('-' * 40)
        print(f'RGB value of Average for id {id} is {average}')
        print('-' * 40)
        # annot changes state of image for some reason!!
        '''
        annot = image.copy()
        annot = cv2.rectangle(annot, (x-rad, y-rad), (x+rad, y+rad),
                            (255,0,0),
                            2
                            )
        list_annots.append(annot)
        #cv2.imshow('annot', annot)
        '''

        color = match_to_closest_color(average)
        print('matched color', color)
        if color is not None:
            color_x_positions[color] = x
        else:
            print(f'ERROR: for the {n} tag, out of {len(ids)} tags,' \
                f' I found no color.')
    '''
    while True:
        # this changes state of image for some reason!!
        for annot_img in list_annots:
            cv2.imshow('annot', annot_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Caught q key, exiting')
                break
        if cv2.waitKey(1) & 0xFF == ord('x'):
            print('Caught x key, exiting')
            break
    '''

    print('here are the output of find_color_xpos', color_x_positions)
    return color_x_positions



def pick_cube_by_color(color):
    color_x_positions = find_color_xpos()
    print(f'found positions: {color_x_positions}')

    slot = find_cube_slot(color, color_x_positions, X_THRES_1, X_THRES_2)

    print(f'found slot {slot} for color {color}')
    try:
        move_to_grab(slot)
        return True
    except:
        print('failed to grab')
        return False

#STATE_PICK = False

if __name__ == '__main__':
    print('inspect camera image')
    # thresholds for three slots
    #x_thres_1, x_thres_2 = inspect_camera_img()
    #find_color_xpos('yellow')
    #color_x_positions = find_color_xpos()
    #slot = find_cube_slot('yellow', color_x_positions, X_THRES_1, X_THRES_2)
    #move_to_grab('left', grab=False)

    FLAG_RUN = False
    FLAG_RUN = True
    if FLAG_RUN:

        filenames = []
        folder = 'speech_output' 

        mapping = {'smarty':0,
                'blue':3,
                'yellow':4,
                'black':5,}

        for phrase in phrases:
            clean_filename = re.sub(r'\W+', '', phrase)
            filename = f'{folder}/{clean_filename}.wav'
            filenames.append(filename)


        q = queue.Queue()

        def callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))

        def find_keyword(voice_result):
            if 'smarty' in voice_result or 'marty' in voice_result or 'smart' in voice_result:
                print('-' * 40)
                print('this is voice result', voice_result)
                print('-' * 40)
                print('keyword recongized')
                idx = 0
                sound_file = filenames[idx]
                os.system(f'play "{sound_file}"')
                q.queue.clear()
                return True
            return False


        def parse_for_phrases(voice_result):
            #if voice_result in self_phrases:
                #return

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
                    idx = mapping[color]
                    sound_file = filenames[idx]
                    os.system(f'play "{sound_file}"')
                    #time.sleep(2)
                    q.queue.clear()
                    #return True
            #return False
            return found_color

            

        my_model = "voice_model"
        device = None

        if not os.path.exists(my_model):
            print ("Please download a model for your language from https://alphacephei.com/vosk/models")
            print ("and unpack as 'model' in the current folder.")

        device_info = sd.query_devices(device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        samplerate = int(device_info['default_samplerate'])

        model = vosk.Model(my_model)
        state_triggered = False

        try:
            with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=device, dtype='int16',
                                    channels=1, callback=callback):
                print('#' * 80)
                print('Press Ctrl+C to stop the recording')
                print('#' * 80)
                print(samplerate, device)

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

