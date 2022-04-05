'''
    When connection to a webcam is established, cvzone.HandTrackingModule will automatically track one hand. 
    
    TO ADD CUSTOM DATASET IMAGES:

        Set your folder_name to whichever directory you desire to add the image datasets to. [EX: '/right' or '/openPalm']
        PATH_TO_SAVE should remain constant as it is the directory that stores the rest of hand gestures directory.
        Press 'c' to start dataset image capture.

        During capture, as right now currentCaptureNumber is first set to 1, the first image will be '1.png'. 
        targetCaptureNumber is set to 2000. Thus after saving images from 1-2000 (after 2000 takes), the last image will be named '2000.png'.

    TEST model.pth:
        This program can also be used to test your newest saved Model parameters named 'model.pth' created through makeModel.ipynb via pressing 't'
'''

import os, sys
os.chdir(sys.path[0])

from cvzone.HandTrackingModule import HandDetector
import cv2 as cv
import numpy as np

import torch
import time
import NeuralNet_DEEP
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
model = NeuralNet_DEEP.NeuralNet()

model.load_state_dict(torch.load("model.pth"))
model.eval()

LABELS = NeuralNet_DEEP.DRONE_LABELS

# ============== Track Gesture ==============

cap = cv.VideoCapture(1)
detector = HandDetector(detectionCon=0.8, maxHands=2)
PADDING = 20


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
            handImg = cv.cvtColor(handImg, cv.COLOR_BGR2GRAY)
            handImg = cv.resize(handImg, (IMG_SIZE, IMG_SIZE))

            handImg = np.array(handImg)

            X = torch.Tensor(handImg).view(-1, 1, IMG_SIZE, IMG_SIZE)
            X = X/255.0 

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
    cv.imshow("Image", img)
    
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