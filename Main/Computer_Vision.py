import cv2 as cv
import mediapipe as mp
import time
from cvzone.HandTrackingModule import HandDetector


# Constants
mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
detector = HandDetector(detectionCon=0.8, maxHands=1)


class DrawComputer:
    def __init__(self):
        #width and height pixels of cv window
        self.width = 500
        self.height = 500

    # Take frame image and manipulate it if needed
    def update_frame(self, frame, droneState, droneBattery):
        dictValue = {}
        
        start = time.perf_counter()
        
        # Resize frame
        frame = cv.resize(frame, (self.width, self.height))
        
        # Any manipulation needed is done here
        if droneState == "facetrack":
            frame, dictValue["bBox"] = self.face_detect(frame)
        elif droneState == "handtrack":
            frame, dictValue["bBox"] = self.hand_detect(frame)
        
        # ========= FPS counter and outputting the image =========
        end = time.perf_counter()
        totalTime = end - start
        
        # Suffering from success
        if totalTime:
            fps = 1 / totalTime
            cv.putText(frame, f'FPS: {int(fps)}', (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv.putText(frame, f'Battery: {int(droneBattery)}', (20,70), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv.imshow("Missile Cam", frame)
        
        return dictValue
    
    def hand_detect(self, frame):
        bBox = []
        hands, frame = detector.findHands(frame)
        
        if hands:
            hand = hands[0]
            bBox = list(hand["bbox"])
            bBox.append(hand["center"])
        
        return frame, bBox
     
    def face_detect(self, frame):
        with mp_facedetector.FaceDetection(model_selection=0, min_detection_confidence=0.60) as face_detection:
            bBox = []
            
            # Convert the BGR frame to RGB
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

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                cv.putText(frame, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        
            return frame, bBox
        
    
    
    
