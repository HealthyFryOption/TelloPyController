from cvzone.HandTrackingModule import HandDetector
import cv2 as cv
import os, sys
import numpy as np

os.chdir(sys.path[0])
import torch
import time
import NeuralNet_DEEP
import importlib

# Reload module if there's any changes to NeuralNet
importlib.reload(NeuralNet_DEEP)

# ============== Save Images ==============

# EVERY HAND SHOULD BE RIGHT HAND

folder_name = "/right"
PATH_TO_SAVE = "./handCapture" + folder_name
captureHandImg = False
currentCaptureNumber = 601
targetCaptureNumber = 1000

# ============== Save Images ==============

# ============== Track Gesture ==============

trackGesture = False

IMG_SIZE = NeuralNet_DEEP.IMG_WIDTH
model = NeuralNet_DEEP.NeuralNet()


# model.load_state_dict(torch.load("./models/testModel(12, 128, 1400)D.pth"))
model.load_state_dict(torch.load("model.pth"))
model.eval()

LABELS = NeuralNet_DEEP.DRONE_LABELS

# ============== Track Gesture ==============

cap = cv.VideoCapture(1)
detector = HandDetector(detectionCon=0.8, maxHands=2)
PADDING = 20


while True:

    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks

    hands= detector.findHands(img, draw=False)  # without draw
    # hands, img = detector.findHands(img)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        startY = max(bbox1[1]-PADDING, 0) 
        endY = max(bbox1[1]+bbox1[3]+PADDING, 0)
        startX = max(bbox1[0]-PADDING-10, 0)
        endX = max(bbox1[0]+bbox1[2]+PADDING, 0)

        # # 25 pixels padding
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
            # cv.imwrite("testPalm1.jpg", handImg)
            # captureHandImg = not captureHandImg
            stop = time.time()

            if not currentCaptureNumber > targetCaptureNumber:
                if stop - start > 0.10:

                    start = time.time()

                    cv.imwrite(PATH_TO_SAVE + f"/{currentCaptureNumber}.png", handImg)

                    currentCaptureNumber += 1
            else:
                print("GOAL REACHED")

        cv.rectangle(img, (startX, startY), (endX, endY), (255,0,0), 2)
    
    # img = cv.resize(img, (900, 750))
    # Display
    cv.imshow("Image", img)
    
    key = cv.waitKey(1) 

    if key == ord('c'):
        captureHandImg = not captureHandImg
        start = time.time() # seconds
        print("CAPTURE MODE:", captureHandImg)

    elif key == ord('q'):
        break
    elif key == ord('t'):
        trackGesture = not trackGesture
        print("TRACK MODE:", trackGesture)
    
cv.destroyAllWindows()