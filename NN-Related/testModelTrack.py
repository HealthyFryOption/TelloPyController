'''
    When connection to a webcam is established, cvzone.HandTrackingModule will automatically track ONE hand. 
    
    TO DO:
    1) To add custom dataset images:

        Set your folder_name to whichever directory you desire to add the image datasets to. [EX: '/right' or '/openPalm']
        PATH_TO_SAVE should remain constant as it is the directory that stores the rest of hand gestures directory.
        Press 'c' to start dataset image capture.

        During capture, currentCaptureNumber is first set to 1, the first image will be '1.png'. 
        targetCaptureNumber is set to 2000. Thus after saving images from 1-2000 (after 2000 takes), the last image will be named '2000.png'.

    2) To test your trained weights:
        This program can also be used to test your newest saved Model parameters named 'model.pth' created through makeModel.ipynb via pressing 't'
'''

import os, sys
os.chdir(sys.path[0])

from cvzone.HandTrackingModule import HandDetector
import cv2 as cv
import numpy as np

import torch
import time
import COV_NET5 as NeuralNet_DEEP
import importlib

# Reload module if there's any changes to NeuralNet_DEEP
importlib.reload(NeuralNet_DEEP)

# ============== Save Images ==============

folder_name = "/right"
PATH_TO_SAVE = "./handCapture" + folder_name
captureHandImg = False
currentCaptureNumber = 1
targetCaptureNumber = 2000

# ============== Save Images ==============

# ============== Track Gesture ==============

trackGesture = False

IMG_SIZE = NeuralNet_DEEP.IMG_WIDTH
model = NeuralNet_DEEP.Neural()

model.load_state_dict(torch.load("model12_D_GHT_5(8,10).pth"))
model.eval()

LABELS = NeuralNet_DEEP.CLASS_LABELS

# ============== Track Gesture ==============

cap = cv.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
PADDING = 15

def his_equ(read_img):
    # convert from BGR color-space to YCrCb
    ycrcb_img = cv.cvtColor(read_img, cv.COLOR_BGR2YCrCb)

    # equalize the histogram on the Y plane
    ycrcb_img[:, :, 0] = cv.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)
    equalized_img = cv.cvtColor(equalized_img, cv.COLOR_BGR2GRAY)

    return equalized_img

print("IMG SIZE:", IMG_SIZE)
while True:

    # Get image frame from webcam
    success, img = cap.read()

    # Find  hand and its landmarks
    hands = detector.findHands(img, draw=False) 

    if hands:
        hand1 = hands[0]
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h

        startY = max(bbox1[1]-PADDING, 0) 
        endY = max(bbox1[1]+bbox1[3]+PADDING, 0)
        startX = max(bbox1[0]-PADDING-10, 0)
        endX = max(bbox1[0]+bbox1[2]+PADDING, 0)

        # 20 pixels padding
        handImg = img[startY:endY, startX:endX]

        if trackGesture:
            cv.imshow("Normal Hand Image", handImg)

            handImg = his_equ(handImg)
            handImg = cv.resize(handImg, (IMG_SIZE, IMG_SIZE))

            cv.imshow("Input Image", handImg)

            X = torch.Tensor(handImg).view(-1, 1, IMG_SIZE, IMG_SIZE)

            X *= (1/255) 

            prediction = model(X)

            answer = -1
            store = prediction[0]

            maxNum = max(store).item()

            # break the probability threshold of 40%
            if(maxNum > 0.4):
                answer = torch.argmax(prediction).item()

            if answer >= 0:
                    label = LABELS[answer]
                    print(label)
            else:
                label = "UNKNOWN"
            per = maxNum*100
            cv.putText(img, f"{label}: {per:.2f}%", (startX, startY - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if captureHandImg:
            stop = time.time()

            if not currentCaptureNumber > targetCaptureNumber:
                if stop - start > 0.10:

                    start = time.time()

                    cv.imwrite(PATH_TO_SAVE + f"/{currentCaptureNumber}.png", handImg)

                    currentCaptureNumber += 1
            else:
                print("GOAL REACHED")

        cv.rectangle(img, (startX, startY), (endX, endY), (255,0,0), 2)
    
    # Display
    cv.imshow("Normal Webcam", cv.resize(img, (800, 650)))
    
    key = cv.waitKey(1) 

    if key == ord('c'):
        captureHandImg = not captureHandImg
        start = time.time()
        print("CAPTURE MODE:", captureHandImg)

    elif key == ord('q'):
        break

    elif key == ord('t'):
        trackGesture = not trackGesture
        print("TRACK MODE:", trackGesture)
    
cv.destroyAllWindows()