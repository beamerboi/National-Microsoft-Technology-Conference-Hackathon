#this is going to be our mask detector feature where you can use it whenever you are going to make a contact with somebody

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime

def maskDetect():
    proto_txt_path = 'deploy.prototxt'
    model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
    face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

    mask_detector = load_model('mask_detector.model')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))

        face_detector.setInput(blob)
        detections = face_detector.forward()

        faces = []
        bbox = []
        results = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                faces.append(face)
                bbox.append((startX, startY, endX, endY))

        if len(faces) > 0:
            results = mask_detector.predict(faces)

        for (face_box, result) in zip(bbox, results):
            (startX, startY, endX, endY) = face_box
            (mask, withoutMask) = result

            label = ""
            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break



#this is going to be our text detection feature where you can use to read anything you want, the fonction will speak out loud so you can 
#hear it clearly

import pytesseract as tess
import cv2
from gtts import gTTS
import os
import random
import playsound

tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image

#img = Image.open('textHand.png')
#text = tess.image_to_string(img)


def textDetect():
# To 3ture Video from webcam
    webcam = cv2.VideoCapture(0)
    # Iterate forever over frames
    while True:

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

        # Display the text
        cv2.imshow('text',  frame)
        key = cv2.waitKey(1)  # We need waitKey to display something

        # Stop if Q key is pressed
        if key == 81 or key == 113:
            break


    # Release the VideoCapture object
    webcam.release()



#this is going to be our color detector feature
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import urllib #for reading image from URL


# construct the argument parse and parse the arguments
def colorDetect():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
        help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
        help="max buffer size")
    args = vars(ap.parse_args())
    
    # define the lower and upper boundaries of the colors in the HSV color space
    lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80), 'black': (64, 64, 64),'purple': (255, 51, 153)} #assign new item lower['blue'] = (93, 10, 0)
    upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255),'black': (150, 150, 150), 'purple': (102, 0, 102)}

    # define standard colors for circle around the object
    colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255), 'black': (0, 0, 0),'purple': (153,0,76)}

    #pts = deque(maxlen=args["buffer"])
    
    # if a video path was not supplied, grab the reference
    # to the webcam
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
        
    
    # otherwise, grab a reference to the video file
    else:
        camera = cv2.VideoCapture(args["video"])
    # keep looping
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if args.get("video") and not grabbed:
            break

        #IP webcam image stream 
        #URL = 'http://10.254.254.102:8080/shot.jpg'
        #urllib.urlretrieve(URL, 'shot1.jpg')
        #frame = cv2.imread('shot1.jpg')

    
        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #for each color in dictionary check object in frame
        for key, value in upper.items():
            # construct a mask for the color from dictionary`1, then perform
            # a series of dilations and erosions to remove any small
            # blobs left in the mask
            kernel = np.ones((9,9),np.uint8)
            mask = cv2.inRange(hsv, lower[key], upper[key])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
            
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
                # only proceed if the radius meets a minimum size. Correct this value for your obect's size
                if radius > 0.5:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                    cv2.putText(frame,key , (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)

        
        # show the frame to our screen
        cv2.imshow("Frame", frame)
        
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
    
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()



#############################################""

#this is the expiry date detector,where we can detect the date of the product and check if it is safe to eat or not
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

def expDate():
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




#this is object detectuib future for now we are only detecting small things
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
def objDetect():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel


#this is the payment feature, where Pixel is going to help you out, we are going to use a payment app API to make the payment,
#currently we have no API from tunisian payment apps, such as D17 or Flouci, that's why we are only going to store the data for the future update.


import cv2
import numpy as np
from pyzbar.pyzbar import decode
barcodeData = ''
def pay():
    
    def decoder(image):
        global barcodeData
        gray_img = cv2.cvtColor(image,0)
        barcode = decode(gray_img)

        for obj in barcode:
            
            points = obj.polygon
            (x,y,w,h) = obj.rect
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 3)

            barcodeData = obj.data.decode("utf-8")
            barcodeType = obj.type
            print(str(barcodeData))
            
            cv2.putText(frame, 'Detected', (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
            print("Barcode: "+barcodeData )
            break

    cap = cv2.VideoCapture(0)
    while str(barcodeData) == '':
        ret, frame = cap.read()
        decoder(frame)
        cv2.imshow('pay', frame)
        code = cv2.waitKey(10)
        
        if code == ord('q'):
            break




# this is the main for our app, this is where the app is going to interacte with you
import speech_recognition as sr # recognise speech
import playsound # to play an audio file
from gtts import gTTS # google text to speech
import random
from time import ctime # get time details
import time
import os # to remove created audio files

class person:
    name = ''
    def setName(self, name):
        self.name = name

def there_exists(terms):
    for term in terms:
        if term in voice_data:
            return True

r = sr.Recognizer() # initialise a recogniser
# listen for audio and convert it to text:
def record_audio(ask=True):
    with sr.Microphone() as source: # microphone as source
        if ask:
            speak('ask')
        audio = r.listen(source)  # listen for the audio via source
        voice_data = ''
        try:
            voice_data = r.recognize_google(audio)  # convert audio to text
        except sr.UnknownValueError: # error: recognizer does not understand
            speak('I did not get that')
        except sr.RequestError:
            speak('Sorry, the service is down') # error: recognizer is not connected
        print(f">> {voice_data.lower()}") # print what user said
        return voice_data.lower()

# get string and make a audio file to be played
def speak(audio_string):
    tts = gTTS(text=audio_string, lang='en') # text to speech(voice)
    r = random.randint(1,20000000)
    audio_file = 'audio' + str(r) + '.mp3'
    print('saving..')
    tts.save(audio_file) # save as mp3
    print('done')
    playsound.playsound(audio_file) # play the audio file
    print(f"Pixel: {audio_string}") # print what app said
    os.remove(audio_file) # remove audio file

def respond(voice_data):
    # 1: greeting
    if there_exists('hey pixel'):
        greetings = [f"hey, how can I help you {person_obj.name}", f"hey, what's up? {person_obj.name}", f"I'm listening {person_obj.name}", f"how can I help you? {person_obj.name}", f"hello {person_obj.name}"]
        greet = greetings[random.randint(0,len(greetings)-1)]
        print(greet)

    # 2: name
    if there_exists(["what is your name","what's your name","tell me your name"]):
        if person_obj.name:
            speak("my name is Pixel")
        else:
            speak("my name is Pixel. what's your name?")

    if there_exists(["my name is"]):
        person_name = voice_data.split("is")[-1].strip()
        speak(f"okay, i will remember that {person_name}")
        person_obj.setName(person_name) # remember name in person object

    # 3: greeting
    if there_exists(["how are you","how are you doing"]):
        speak(f"I'm very well, thanks for asking {person_obj.name}")

    # 4: time
    if there_exists(["what's the time","tell me the time","what time is it"]):
        time = ctime().split(" ")[3].split(":")[0:2]
        if time[0] == "00":
            hours = '12'
        else:
            hours = time[0]
        minutes = time[1]
        time = f'{hours} {minutes}'
        speak(time)
    
    if there_exists('pay'):
        pay()
    elif there_exists('object detect'):
        objDetect()
    elif there_exists('expiration date'):
        expDate()
    elif there_exists('color'):
        colorDetect()
    elif there_exists('read'):
        textDetect()
    elif there_exists('contact'):
        maskDetect()


    if there_exists(["exit", "quit", "goodbye"]):
        speak("going offline")
        exit()
    


time.sleep(1)


person_obj = person()
while(1):
    voice_data = record_audio() # get the voice input
    respond(voice_data) # respond

