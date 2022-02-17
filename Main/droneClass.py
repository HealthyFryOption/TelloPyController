from djitellopy import tello
from cv2 import imwrite, VideoWriter, VideoWriter_fourcc
import os
from datetime import datetime
from time import sleep



class Drone:
    '''
        DRONE OBJECT THAT CONTAINS ALL DRONE CONTROLS AND INITIALIZATION OF CONNECTION
        STATE： [keycontrols, facetrack]
    '''
    
    def __init__(self):
        # ======== Initialise Drone ============
        # Drone object
        self.drone = tello.Tello()
        self.drone.connect()
        self.drone.streamon()

        # Drone movement controls
        # lr, fb, ud, yv [list order]
        self.rc_controls = [0,0,0,0]
        
        # State of drone
        self.state = "keycontrols"
        
        # Recording State of drone
        self.videoCapState = False
        
        # Current frame captured by drone
        self.current_frame = None
        
        #used to verify that drone is connected
        print(self.drone.get_battery())
    
    # All updates needed
    def update_drone(self):
        self.drone.send_rc_control(*self.rc_controls)
    def land_drone(self):
        self.drone.land()
    def takeoff_drone(self):
        self.drone.takeoff()
    def get_battery(self):
        print(self.drone.get_battery())
    def flip_drone(self, direction):
        self.drone.flip(direction)
    
    # Use set current frame as picture in specified folder
    def take_picture(self):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%H-%M-%S")
        path = os.getcwd() + "/images"
        
        pic_name = "picDrone-" + str(dt_string) +".jpg"
        imwrite(os.path.join(path, pic_name) , self.current_frame)
        print("Picture Taken")
    
   