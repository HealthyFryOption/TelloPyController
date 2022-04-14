'''
    ===================
    AUTHORED BY 

    LorenzoPixel
    CrazyDud22
    ===================
    
    main.py is the only file to run. Contains the main while loop runner for constant connection and manipulation to the Tello Drone.
    The program is recommended to run in Python 3.9 and above.


    LATEST VERSION OF ESSENTIAL MODULES DURING TEST RUNS
    PyGame - 2.0.1
    opencv-python - 4.5.3.56
    mediapipe - 0.8.7.3
    cvzone - 1.5.6
    torch - 1.11.0
    numpy - 1.21.2
'''

import os
from sys import path as sysPath
os.chdir(sysPath[0])
print("CWD: " + os.getcwd())

import controls
import droneClass
import Computer_Vision
import track_prop

import time
import traceback
from threading import Thread
from datetime import datetime
from cv2 import VideoWriter, VideoWriter_fourcc, destroyAllWindows


# Get needed initialization of objects
cv_obj = Computer_Vision.DrawComputer()
drone_obj = droneClass.Drone()
tracking_prop = track_prop.TrackP()

# Constants
cv_display_width = cv_obj.width
cv_display_height = cv_obj.height

is_record = False

def videoRecorder():
    # create a VideoWrite object and save recording to ./video.avi

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%H-%M-%S")
    path = os.getcwd() + "/videos"
    vid_name = "vidDrone-" + str(dt_string) +".avi"
    
    height, width, _ = drone_obj.current_frame.shape
    video = VideoWriter(os.path.join(path, vid_name), VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while drone_obj.videoCapState:
        video.write(drone_obj.current_frame)

        # 30 FPS Video
        time.sleep(1 / 30)

    video.release()

def main():

    try:

        # drone_obj.running set to False only if PG window is closed [see controls.py for more info]
        while drone_obj.running:
            
            # Get frame image from Drone's camera
            drone_obj.current_frame = drone_obj.drone.get_frame_read().frame 
            

            # =========== Recording ==========
            if drone_obj.videoCapState and not is_record:
                record = Thread(target=videoRecorder)
                record.start()
                is_record = True

            elif not drone_obj.videoCapState:
                is_record = False
            # =========== Recording ===========
            

            # Get any keys pressed in PyGame window and change drone object's states accordingly
            controls.keyboard_controls(drone_obj, tracking_prop)
            
            # Output captured frame into CV Window and get necessary additional outputs in the form of a dictionary.
            dictValues = cv_obj.update_frame(drone_obj.current_frame, drone_obj.state, drone_obj.get_battery())
            

            # Control Movement for different states other than keyboard controls
            if drone_obj.state in ["facetrack", "handtrack"]:
                
                # Whether the boundaryBox is an empty iteratable or have coordinates, track_movement() will check on its own and set movement accordingly
                controls.track_movement(drone_obj, tracking_prop, dictValues["bBox"], (cv_display_width, cv_display_height))   
            
            elif drone_obj.state == "gesture":
                controls.gesture_movement(drone_obj, dictValues["label"])

            print(drone_obj.state)
            
            # Update drone to move according to new RC Controls set
            drone_obj.update_drone()
            
            # Sleep to prevent overloading PyGame as it will if there's no time in between for keyboard event listening
            time.sleep(0.02)
            
    except Exception as e:

        # Land drone if an exception in the code is found
        drone_obj.land_drone()
        drone_obj.running = False 
        
        print("======================\nException Happened: ", str(e.__class__), f"\n{e}\n")
        print(traceback.format_exc(), "\n======================")
    
    finally:
        # Destroy all CV windows
        destroyAllWindows()

if __name__ == "__main__" :
    main()