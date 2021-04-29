import pytesseract as tess
import cv2
from gtts import gTTS
import os
import random
import playsound


tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image


webcam = cv2.VideoCapture(0)
# Iterate forever over frames
while True:
    ret, frame = webcam.read()
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    text = tess.image_to_string(frame)
    try:
        tts = gTTS(text=text, lang='en')
        r = random.randint(1,20000000)
        audio_file = 'audio' + str(r) + '.mp3'
        tts.save(audio_file) 
        playsound.playsound(audio_file)
        os.remove(audio_file) 

    except:
        print(text)
    cv2.imshow('Text Detection', frame)
    code = cv2.waitKey(10)
    
    if code == 81 or code == 113:
        break


# Release the VideoCapture object
webcam.release()