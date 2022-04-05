# Tello Py Controller

## Introduction
Tello Py Controller can be used to control your Tello Ryze drone in **4 modes**:

* Keyboard Control
  - Enables various movement and gfeature control of your Drone via your computer's keyboard. More info can be found inside controls.py
* Facial Tracking
  - Constantly look for a face with the highest detection probability inside the image frame captured by the drone.
  - Through PID Controllers set inside controls.py, follows the face around.
* Hand Tracking
  - Constantly look for a hand with the highest detection probability inside the image frame captured by the drone.
  - Follows the hand vertically and horizontally only. The bigger the hand detected, resulting in a larger boundary box, the further the drone will move away. Vice versa, the smaller the hand detected, the closer it will move forward.
* Gesture Recognition Movement
  -  Constantly look for a hand with the highest detection probability inside the image frame captured by the drone.
  -  Parses the section of the frame containing the detected hand and pass it to an original **Convolutional Neural Network Model** which classifies what gesture it is and subsequently manipulate the drone's movement.

Thus repository is comprised of two directory, **Main** and **NN-Related**.

## Main
This directory contains all the necessary Python files needed for you to run a connection to your Tello Ryze drone and manipulate its movement.

A few notes to take into account:

* You are needed to only ever needed to run main.py
* Computer_Vision.py majorly handles the drawing or classifications of frames/images, and subsequently provide necessary boundary box information. controls.py then takes in given information, especially boundary boxes around object detected to control the drone's movement. 
* The directory **GestureClassifier** is used to store the Convolutional Neural Network architecture written in PyTorch and a trained model's saved parameters, preferably in type .pth . We have not provided the parameters as it is about the storage space limit of 25mb. Thus, we encourage you to build your own via NN-Related!


