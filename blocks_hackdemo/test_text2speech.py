'''
27 Dec 2021
author: nouyang
'''
import os
import re

'''
TEXT='Hi!'
tts --text $TEXT --out_path tts_$NAME.wav; aplay tts_$NAME.wav
'''

# https://github.com/mozilla/TTS/tree/e9e07844b77a43fb0864354791fb4cf72ffded11/TTS/server

phrases = ["Hi there!", 
           "Say that again?",
           "Okay, I can do that",
           "Okay, getting the blue cube",
           "Okay, getting the yellow cube",
           "Okay, getting the black cube"]

def generate_speech(phrases, preview=False, overwrite_files=False):
    folder = 'speech_output' 
    # -- Make folder if not exist
    if not os.path.exists(folder):
        print(f'Making directory: {folder}\n')
        os.makedirs(folder)

    print('Generating speech')
    for phrase in phrases:

        clean_filename = re.sub(r'\W+', '', phrase)
        filename = f'{folder}/{clean_filename}.wav'

        file_exists = (os.path.exists(filename) and os.stat(filename).st_size != 0)
        if file_exists and not overwrite_files:
            print(f'- Phrase already exists: {phrase}')
        else:
            #os.system('MODEL="tts_models/en/ljspeech/glow-tts"')
            #os.system('tts --text "What would you like me to do?" --model_name $MODEL')
            os.system(f'tts --text "{phrase}" --out_path "{filename}"')
        if preview:
            os.system(f'play "{filename}"')

generate_speech(phrases, preview=True)
