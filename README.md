# TelloPyController

This repository is made to complete our Diploma course.

![alt text](https://www.newegg.com/insider/wp-content/uploads/2018/03/DJI-Tello-03257-1-1024x576.jpg)

## Introduction
TelloPyController can be used to control your Tello Ryze drone in **4 modes**:

* Keyboard Control
  - Enables various movement and feature control of your Drone via your computer's keyboard. More info can be found inside controls.py
* Facial Tracking
  - Constantly look for a face with the highest detection probability inside the image frame captured by the drone.
  - Through PID Controllers set inside controls.py, follows the face around.
* Hand Tracking
  - Constantly look for a hand with the highest detection probability inside the image frame captured by the drone.
  - Follows the hand vertically and horizontally only. The bigger the hand detected, the larger its boundary box. Thus, the further the drone will move backwards. Vice versa, the smaller the hand detected, the smaller its boundary box, and the closer it will move forward.
* Gesture Recognition Movement
  -  Constantly look for a hand with the highest detection probability inside the image frame captured by the drone.
  -  Parses the section of the image frame containing the detected hand and pass it to an original **Convolutional Neural Network Model** which classifies what gesture it is and subsequently manipulate the drone's movement.

Picture and video taking capabilities is also enabled.

This repository is comprised of two directory, **Main** and **NN-Related**.

## Main
This directory contains all the necessary Python files needed for you to run a connection to your Tello Ryze drone and manipulate its movement, in various modes and with various features as mentioned previously.

A few notes to take into account:

* You are needed to only ever needed to run main.py
* Computer_Vision.py majorly handles the drawing or classifications of frames/images, and subsequently provide necessary information. controls.py then takes in given information, such as boundary boxes around object detected or label of gesture detected to control the drone's movement. 
* To stop execution, close the PyGame window invoked only.
* The directory **GestureClassifier** is used to store the Convolutional Neural Network architecture written in PyTorch and a trained model's saved parameters, preferably in type .pth . We have not provided the parameters as it is about the storage space limit of 25mb. Thus, we encourage you to build your own via NN-Related!

For more information as to how the code works, and where to customized certain values, please refer to the docstrings or comments available.

## NN-Related
This directory contails all the necessary Python files needed for you to *create* your .pth file to be placed inside Main.GestureClassifier. You can either use the already provided Convolutional Neural Network architecture in PyTorch and train it, or customize it to your own configurations.

**For more information on how to use it, please check SUMMARY.txt available inside it. You can also refer to the readily prepared docstrings documentation for more information**

## DISCLAIMER
If you plan to use this repository for your own drone's test run, make sure that there are little to no WiFi-interference as it may cause bad connection between the device running the program and your drone. This will then ensue corruption of frame captured from the drone which could cause issues in certain modes. Furthermore, make sure the climate around is not too windy so that a smooth flight can be taken.

**The authors and contributors of this repository is NOT liable for ANY kinds of damages that may incur. Run at your own discretion.**

## REFERENCES
The NN-Related directory was created with reference from YouTuber sentdex with our own modifications. Special thanks to sentdex for helping kickstart our PyTorch self-studies.

Channel: https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
