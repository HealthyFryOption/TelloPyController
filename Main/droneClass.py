'''
    Contains class Drone whose object help send all drone controls and maintain connection to the Tello Drone.
    State: [keycontrols, facetrack, handtrack, gesture]
'''

from djitellopy import tello
from cv2 import imwrite
import os
from datetime import datetime

class Drone:
    
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
        
        # Verify that drone is connected
        print(self.drone.get_battery())

        # Flying and running state of drone

        self.running = True # Only manipulated in the outside
        self.flying = False 
    
    # Each iteration of the main loop will call update_drone to give new movement commands to drone via manipulation of rc_controls list
    def update_drone(self):
        self.drone.send_rc_control(*self.rc_controls)

    def land_drone(self):
        self.drone.land()
        self.flying = False

    def takeoff_drone(self):
        self.drone.takeoff()
        self.flying = True

    def get_battery(self):
        return self.drone.get_battery()

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