'''
27 Dec 2021
author: nouyang
'''
import os
os.system('TTS="tts_models/en/ljspeech/glow-tts"')
os.system('tts --text "What would you like me to do?" --model_name $TTS')
os.system('play tts_output.wav')
# https://github.com/mozilla/TTS/tree/e9e07844b77a43fb0864354791fb4cf72ffded11/TTS/server

'''
from playsound import playsound
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager



synth = Synthesizer(
wavs = synth.tts('this is a test')
#out = io.BytesIO()
synth.save_wav(wavs, 'output_wav.wav')
playsound('tts_output.wav')

# Run the server with the official models. python TTS/server/server.py --model_name tts_models/en/ljspeech/tacotron2-DCA --vocoder_name vocoder_models/en/ljspeech/mulitband-melgan

'''

'''
# Generates TTS using espeak, sounds terrible, but it's fast and offline

import pyttsx

engine = pyttsx.init()
voices = engine.getProperty('voices')
print(len(voices))

for i in range(len(voices)):
    print(f'Voice {i} out of {len(voices)})
    engine.setProperty('voice', voices[i].id) #change index to change voices
    engine.say(f'Voice {i}: Which cube would you like?')
    engine.runAndWait()
'''
