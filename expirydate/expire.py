
from pytesseract import image_to_string
import pytesseract as tess
import cv2
from datetime import datetime
import calendar
import time
from gtts import gTTS
import os
import random
import playsound

tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

webcam = cv2.VideoCapture(0)

while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()

    text = tess.image_to_string(frame)
    try:
        date_time_str = text

        date_time_obj = datetime.strptime(date_time_str, '%d/%m/%y')
        #we wanted to make the second date but we couldn't figure how we can do it
        t1 = calendar.timegm(time.strptime(date_time_obj, '%d, %m, %Y UTC'))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        tn = calendar.timegm(time.strptime(current_time.now()))
        if tn > t1:
            out = "Out of date"
        elif tn <= t1:
            out = "Enjoy eating"
    except:
        out = "Move"

    tts = gTTS(text=out, lang='en')
    r = random.randint(1,20000000)
    audio_file = 'audio' + str(r) + '.mp3'
    tts.save(audio_file) 
    playsound.playsound(audio_file)
    os.remove(audio_file) 

    ret, frame = webcam.read()
    cv2.imshow('pay', frame)
    code = cv2.waitKey(10)
    
    if code == ord('q'):
        break
#cv2.imshow("Cropped",  crop_img)
