'''
27 Dec 2021
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

filenames = []
folder = 'speech_output' 

mapping = {'blue':3,
           'yellow':4,
           'black':5}

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

def parse_for_phrases(voice_result):
    #if voice_result in self_phrases:
        #return

    print('-' * 40)
    print('this is voice result', voice_result)
    print('-' * 40)
    colors = ['blue', 'yellow', 'black']
    words = voice_result.split(' ')
    for color in colors:
        if color in words:
            print(f'Parsed {color}!')
            idx = mapping[color]
            sound_file = filenames[idx]
            os.system(f'play "{sound_file}"')
            #time.sleep(2)
            q.queue.clear()

my_model = "voice_model"
device = None

if not os.path.exists(my_model):
    print ("Please download a model for your language from https://alphacephei.com/vosk/models")
    print ("and unpack as 'model' in the current folder.")

device_info = sd.query_devices(device, 'input')
# soundfile expects an int, sounddevice provides a float:
samplerate = int(device_info['default_samplerate'])

model = vosk.Model(my_model)

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
                    sentence = rec.Result()
                    parse_for_phrases(sentence)
                else:
                    print(rec.PartialResult())


except KeyboardInterrupt:
    print('\nDone')
except Exception as e:
    print('Exception: ', e)
