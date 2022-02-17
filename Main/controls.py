import pygame as pg
import cv2
import numpy as np

# Constants
speed = 100

# The central point is 9000 area in pixel units^2, with margin error of 2000 area pixel units^2 
front_back_range = [10000, 16000]
b_intended_area = 13000

# Proportion Integral Derivative Controller for facial. [0] -> kp, [1] -> kd, [2] -> ki
pid = [0.41, 0.41, 0]
fb_pid = [0.0030, 0.0030, 0]

# Proportion Integral Derivative Controller for hand. [0] -> kp, [1] -> kd, [2] -> ki
h_ud_pid = [0.30, 0.30, 0]
h_lr_pid = [0.15, 0.15, 0]
h_fb_pid = [0.0025, 0.0025, 0]

#Display W/H
pg_display_width = 300
pg_display_height = 300

pg.init()
pgWindow = pg.display.set_mode((pg_display_width, pg_display_height))
pg.display.set_caption('Controls')

# Drone movement controls
# lr, fb, ud, yv [list order]

movements = {
    "left" : -speed,
    "right" : speed,
    "up" : speed,
    "down" : -speed,
    "yaw_left" : -speed,
    "yaw_right" : speed,
    "front" : speed,
    "back" : -speed,
}

def keyboard_controls(drone, track_prop):
    events = pg.event.get()
    for event in events:
        if event.type == pg.QUIT:
            cv2.destroyAllWindows()
            pg.quit()
            
        # If a key is detected to be pressed down
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_1:
                drone.state = "keycontrols"
                drone.rc_controls = [0,0,0,0]
                
                track_prop.h_pError = 0
                track_prop.v_pError = 0
                track_prop.b_pError = 0
                
            elif event.key == pg.K_2:
                drone.state = "facetrack"
                drone.rc_controls = [0,0,0,0]
                
                track_prop.h_pError = 0
                track_prop.v_pError = 0
                track_prop.b_pError = 0
            
            elif event.key == pg.K_3:
                drone.state = "handtrack"
                drone.rc_controls = [0,0,0,0]
                
                track_prop.h_pError = 0
                track_prop.v_pError = 0
                track_prop.b_pError = 0
                
            elif event.key == pg.K_q:
                drone.takeoff_drone()
            elif event.key == pg.K_e:
                drone.land_drone()
                print(drone.get_battery())
            elif event.key == pg.K_c:
                drone.take_picture()
            elif event.key == pg.K_z:
                drone.videoCapState = True
                print("Drone is Recording")
                
            elif event.key == pg.K_x:
                drone.videoCapState = False
                print("Drone is not / stopped Recording")
                
                
            # elif event.key == pg.K_t:
            #     drone.drone.initiate_throw_takeoff()
                
            if drone.state == "keycontrols":
                if event.key == pg.K_a:
                    drone.rc_controls[3] = movements["yaw_left"]
                elif event.key == pg.K_d:
                    drone.rc_controls[3]  = movements["yaw_right"]
                elif event.key == pg.K_LEFT:
                    drone.rc_controls[0]  =  movements["left"]
                elif event.key == pg.K_RIGHT:
                    drone.rc_controls[0] = movements["right"]
                elif event.key == pg.K_UP:
                    drone.rc_controls[1] = movements["front"]
                elif event.key == pg.K_DOWN:
                    drone.rc_controls[1] =  movements["back"]
                elif event.key == pg.K_w:
                    drone.rc_controls[2] = movements["up"]
                elif event.key == pg.K_s:
                    drone.rc_controls[2] =  movements["down"]
                elif event.key == pg.K_i:
                    drone.flip_drone("f")
                elif event.key == pg.K_j:
                    drone.flip_drone("l")
                elif event.key == pg.K_k:
                    drone.flip_drone("b")
                elif event.key == pg.K_l:
                    drone.flip_drone("r")
              
    
        # If a key is detected to be lifted up
        elif event.type == pg.KEYUP:
            if drone.state == "keycontrols":
                if event.key == pg.K_a:
                    if drone.rc_controls[3] == movements["yaw_left"]:
                        drone.rc_controls[3] = 0
                elif event.key == pg.K_d:
                    if drone.rc_controls[3] == movements["yaw_right"]:
                        drone.rc_controls[3] = 0
                elif event.key == pg.K_LEFT:
                    if drone.rc_controls[0] == movements["left"]:
                        drone.rc_controls[0] = 0
                elif event.key == pg.K_RIGHT:
                    if drone.rc_controls[0] == movements["right"]:
                        drone.rc_controls[0] = 0
                elif event.key == pg.K_UP:
                    if drone.rc_controls[1] == movements["front"]:
                        drone.rc_controls[1] = 0
                elif event.key == pg.K_DOWN:
                    if drone.rc_controls[1] == movements["back"]:
                        drone.rc_controls[1] = 0
                elif event.key == pg.K_w:
                    if drone.rc_controls[2] == movements["up"]:
                        drone.rc_controls[2] = 0
                elif event.key == pg.K_s:
                    if drone.rc_controls[2] == movements["down"]:
                        drone.rc_controls[2] = 0

def track_movement(drone, tracking_prop, bBox, cv_display):
    # If no face is tracked, don't run cuz attribute would not be found
    if bBox:     
        # Constants
        cv_display_width = cv_display[0]
        cv_display_height = cv_display[1]
        
        if drone.state == "facetrack":
            box_width = bBox.width * cv_display_width
            box_height = bBox.height * cv_display_height
            
            box_half_width = box_width // 2
            box_half_height = box_height // 2
            
            cx = (bBox.xmin * cv_display_width) + box_half_width
            cy = (bBox.ymin * cv_display_height) + box_half_height
            
            box_area = box_width * box_height
        
            # === Front Back Control ===
            
            if box_area > front_back_range[0] and box_area < front_back_range[1]:
                drone.rc_controls[1] = 0
                b_error = 0
            else:
                b_error = box_area - b_intended_area
                
                # kp * error + kd * (error- pError)
                speed = fb_pid[0]*b_error + fb_pid[1]*(b_error - tracking_prop.b_pError)
                speed = int(np.clip(speed, -45, 45))

                drone.rc_controls[1] = -speed 
            
            # === Front Back Control ===
            
            
            # === Horizontal Yawing Control ===     
            cx = (bBox.xmin * cv_display_width) + box_half_width
            
            h_error = cx - (cv_display_width // 2)

            # kp * error + kd * (error- pError)
            speed = pid[0]*h_error + pid[1]*(h_error- tracking_prop.h_pError)
            speed = int(np.clip(speed, -100, 100))

            drone.rc_controls[3] = speed    
            # === Horizontal Yawing Control ===
            
            
            # === Vertical Control ===     
            cy = (bBox.ymin * cv_display_height) + box_half_height
            
            v_error = cy - (cv_display_height // 2)

            # kp * error + kd * (error- pError)
            speed = pid[0]*v_error + pid[1]*(v_error- tracking_prop.v_pError)
            speed = int(np.clip(speed, -35, 35))

            drone.rc_controls[2] = -speed
            print("vertical speed: " + str(speed))    
            # === Vertical Control ===
        
        elif drone.state == "handtrack":
            box_width = bBox[2]
            box_height = bBox[3]
            
            box_half_width = box_width // 2
            box_half_height = box_height // 2
            
            cx = bBox[4][0]
            cy = bBox[4][1]
        
            box_area = box_width * box_height
        
            # === Front Back Control ===
            if box_area > front_back_range[0] and box_area < front_back_range[1]:
                drone.rc_controls[1] = 0
                b_error = 0
            else:
                b_error = box_area - b_intended_area
                
                # kp * error + kd * (error- pError)
                speed = h_fb_pid[0]*b_error + h_fb_pid[1]*(b_error - tracking_prop.b_pError)
                speed = int(np.clip(speed, -45, 45))

                drone.rc_controls[1] = -speed 
            
            # === Front Back Control ===
            
            
            # === Horizontal Yawing Control ===     
            h_error = cx - (cv_display_width // 2)

            # kp * error + kd * (error- pError)
            speed = h_lr_pid[0]*h_error + h_lr_pid[1]*(h_error- tracking_prop.h_pError)
            speed = int(np.clip(speed, -40, 40))

            drone.rc_controls[0] = speed    
            # === Horizontal Yawing Control ===
            
            
            # === Vertical Control ===               
            v_error = cy - (cv_display_height // 2)

            # kp * error + kd * (error- pError)
            speed = h_ud_pid [0]*v_error + h_ud_pid [1]*(v_error- tracking_prop.v_pError)
            speed = int(np.clip(speed, -35, 35))

            drone.rc_controls[2] = -speed
            print("vertical speed: " + str(speed))    
            # === Vertical Control ===

    else:
        drone.rc_controls[0] = 0
        drone.rc_controls[1] = 0
        drone.rc_controls[2] = 0    
        drone.rc_controls[3] = 0
        h_error = 0
        v_error = 0
        b_error = 0
    
    # give back error to main.py as previous error
    tracking_prop.h_pError, tracking_prop.v_pError, tracking_prop.b_pError = h_error, v_error, b_error
        
       
            
            
        