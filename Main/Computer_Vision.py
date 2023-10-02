'''
    Contains class DrawComputer whose object is responsible for 3 purposes:

    1) Drawing and updating needed information on captured frames to be displayed on the main CV Window
    2) Perform object detection of hands, gestures, and faces.
    3) Help retrieve the coordinates of detected objects' boundary box which will be given to controls.py


    GestureClassifier contains a Convolutional Neural Network architecture in PyTorch and a trained model's learned parameters through it in the form of a .pth extension.
    see [ https://pytorch.org/tutorials/beginner/saving_loading_models.html ] for more info
'''

import importlib
import GestureClassifier.COV_NET5 as NND
import torch
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from time import perf_counter
import mediapipe as mp
import cv2 as cv
import os
from sys import path as sysPath

os.chdir(sysPath[0])


# Constants
mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
detector = HandDetector(detectionCon=0.8, maxHands=1)

# HAND GESTURE MODEL

# Reload module if there's any changes to the CNN's architecture used
importlib.reload(NND)

IMG_WIDTH = NND.IMG_WIDTH
IMG_HEIGHT = NND.IMG_HEIGHT

model = NND.Neural()

model.load_state_dict(torch.load("./GestureClassifier/ModelToUse.pth",
                      map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
model.eval()

LABELS = NND.CLASS_LABELS

# HAND GESTURE MODEL


class DrawComputer:

    PADDING = 15  # padding to get hand gesture image in parseGesture()

    def __init__(self):
        # Constant width and height in pixels of cv window
        self.width = 500
        self.height = 500

    def parseGesture(self, bBox, frame):

        startX = max(bBox[0]-self.PADDING-10, 0)
        endX = max(bBox[0]+bBox[2]+self.PADDING, 0)

        startY = max(bBox[1]-self.PADDING, 0)
        endY = max(bBox[1]+bBox[3]+self.PADDING, 0)

        handImg = frame[startY:endY, startX:endX]

        return handImg, [startX, startY, endX, endY]

    # convert to GRAYSCALED Histogram Equalized
    def his_equ(self, read_img):
        # convert from BGR color-space to YCrCb
        ycrcb_img = cv.cvtColor(read_img, cv.COLOR_BGR2YCrCb)

        # equalize the histogram on the Y plane
        ycrcb_img[:, :, 0] = cv.equalizeHist(ycrcb_img[:, :, 0])

        # convert back to RGB color-space from YCrCb
        equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)
        equalized_img = cv.cvtColor(equalized_img, cv.COLOR_BGR2GRAY)

        return equalized_img

    # Take frame image and manipulate it if needed
    def update_frame(self, frame, droneState, droneBattery):
        dictValue = {}

        # START_TIMER
        start = perf_counter()

        # Resize frame [need to take frame output]
        frame = cv.resize(frame, (self.width, self.height))

        # Any manipulation needed is done here based on state of drone
        if droneState == "facetrack":
            frame, dictValue["bBox"] = self.face_detect(frame)

        elif droneState == "handtrack":
            frame, dictValue["bBox"] = self.hand_detect(frame)

        elif droneState == "gesture":
            dictValue["bBox"] = self.hand_detect_for_gesture(frame)
            dictValue["label"] = ""  # initialise as empty string

            # if any hands were detected at all
            if dictValue["bBox"]:
                handImg, newBbox = self.parseGesture(dictValue["bBox"], frame)

                # replace empty string with the name of the label detected from model
                # Frame, HandImg, newBbox
                dictValue["label"] = self.gesture_detect(
                    frame, handImg, newBbox)

        # ========= FPS counter and outputting the image =========
        end = perf_counter()
        totalTime = end - start

        if totalTime:
            fps = 1 / totalTime
            cv.putText(frame, f'FPS: {int(fps)}', (20, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv.putText(frame, f'Battery: {int(droneBattery)}',
                   (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if droneState == "keycontrols":
            cv.putText(frame, f'Mode: Keyboard', (20, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif droneState == "facetrack":
            cv.putText(frame, f'Mode: Face Tracking', (20, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif droneState == "handtrack":
            cv.putText(frame, f'Mode: Hand Tracking', (20, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif droneState == "gesture":
            cv.putText(frame, f'Mode: Gesture Tracking', (20, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv.imshow("Drone Camera", frame)

        return dictValue

    def hand_detect(self, frame):
        bBox = []
        hands, frame = detector.findHands(frame)

        if hands:
            hand = hands[0]
            bBox = list(hand["bbox"])
            bBox.append(hand["center"])

        return frame, bBox

    def hand_detect_for_gesture(self, frame):
        bBox = []
        hands = detector.findHands(frame, draw=False)

        if hands:
            hand = hands[0]
            bBox = list(hand["bbox"])

        return bBox

    def gesture_detect(self, frame, handImg, newBbox):

        # Turn it GRAYSCALE as model accepts 1 channel input only
        # handImg = cv.cvtColor(handImg, cv.COLOR_BGR2GRAY)

        handImg = self.his_equ(handImg)

        # Resize it to what has been standardized
        handImg = cv.resize(handImg, (IMG_HEIGHT, IMG_WIDTH))

        # Flatten the data
        X = torch.Tensor(handImg).view(-1, 1, IMG_HEIGHT, IMG_WIDTH)

        # Turn pixel values into decimals and into range 0-1 normalization
        X *= (1/255)

        # Place input inside model to get prediction
        prediction = model(X)

        answer = -1
        store = prediction[0]

        maxNum = max(store).item()

        label = "UNKNOWN"

        # If probability of detection is more than the probability here
        if (maxNum > 0.5):
            answer = torch.argmax(prediction).item()

        if answer >= 0:
            label = LABELS[answer]

        # Display current probability
        per = maxNum*100

        cv.putText(frame, f"{label}: {per:.2f}%", (20, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv.rectangle(frame, (newBbox[0], newBbox[1]),
                     (newBbox[2], newBbox[3]), (255, 0, 0), 2)

        return label

    def face_detect(self, frame):
        with mp_facedetector.FaceDetection(model_selection=0, min_detection_confidence=0.60) as face_detection:
            bBox = []

            # Convert BGR frame to RGB
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process the frame and find faces
            results = face_detection.process(frame)

            # Convert the frame color back so it can be displayed
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            if results.detections:
                detection = results.detections[0]

                # Draw face detection pointers
                mp_draw.draw_detection(frame, detection)
                bBox = detection.location_data.relative_bounding_box

                h, w, c = frame.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin *
                                                   h), int(bBox.width * w), int(bBox.height * h)

                cv.putText(frame, f'{int(detection.score[0]*100)}%', (boundBox[0],
                           boundBox[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            return frame, bBox
