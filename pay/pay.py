import cv2
import numpy as np
from pyzbar.pyzbar import decode


barcodeData = ''
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

        barcodeType = obj.type
        
        if barcodeType != "QRCODE":
            continue
        else:
            barcodeData = obj.data.decode("utf-8")
            cv2.putText(frame, 'Detected', (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
            # it should be sent through a paymenet api here
            print("Barcode: "+barcodeData +" | Type: "+barcodeType)
            break

cap = cv2.VideoCapture(0)
while str(barcodeData) == '':
    ret, frame = cap.read()
    decoder(frame)
    cv2.imshow('pay', frame)
    code = cv2.waitKey(10)
    
    if code == ord('q'):
        break


