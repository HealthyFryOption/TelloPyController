import controls
import droneClass
import Computer_Vision
import time
import track_prop
import sys
import os
from threading import Thread
from datetime import datetime
from cv2 import VideoWriter, VideoWriter_fourcc

os.chdir(sys.path[0])
print("CWD: " + os.getcwd())

# Get needed initialization of objects
cv_obj = Computer_Vision.DrawComputer()
drone_obj = droneClass.Drone()
tracking_prop = track_prop.TrackP()

# Constants
cv_display_width = cv_obj.width
cv_display_height = cv_obj.height
is_record = False

def videoRecorder():
    # create a VideoWrite object, recoring to ./video.avi
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%H-%M-%S")
    path = os.getcwd() + "/videos"
    vid_name = "vidDrone-" + str(dt_string) +".avi"
    
    height, width, _ = drone_obj.current_frame.shape
    video = VideoWriter(os.path.join(path, vid_name), VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while drone_obj.videoCapState:
        video.write(drone_obj.current_frame)
        time.sleep(1 / 30)

    video.release()

try:
    while True:
        # Get frame image from Drone's camera
        drone_obj.current_frame = drone_obj.drone.get_frame_read().frame 
        
        # If recorded 
        if drone_obj.videoCapState and not is_record:
            record = Thread(target=videoRecorder)
            record.start()
            is_record = True
            
        elif not drone_obj.videoCapState:
            is_record = False
        
        # Get any keys pressed in Pygame window
        controls.keyboard_controls(drone_obj, tracking_prop)
        
        # Output image into CV Window and get additional outputs
        dictValues = cv_obj.update_frame(drone_obj.current_frame, drone_obj.state, drone_obj.get_battery())
        
        if drone_obj.state in ["facetrack", "handtrack"]:
            box = dictValues["bBox"]

            controls.track_movement(drone_obj, tracking_prop, box, (cv_display_width, cv_display_height))
        
        print(drone_obj.state)
        drone_obj.update_drone()
              
        # Sleep to prevent overloading PyGame as data overloads if there's no time in between for keyboard event listening
        time.sleep(0.02)
        
except Exception as e:
    print("======================\nException Happened: " + str(e.__class__) + f"\n{e}")
    drone_obj.land_drone()

# All windows are closed when pygame window quits


