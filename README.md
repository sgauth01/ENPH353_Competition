# ENPH353_Competition

In this project, we developed the control code for a robot navigating in a Gazebo simulated environment. The goal was to implement an autonomous agent capable of driving through the environment while obeying traffic laws and returning license plates of parked cars and associated parking IDs. The only input we receive from the environment is real-time images from the in-simulation camera view from the robot. We have two different ROS nodes that process these images. One of these nodes publishes velocity commands to move the robot around the environment, while the other publishes license plate numbers and locations to a license plate tracker application. These nodes run in parallel to ensure the robot is driving and detecting license plates simultaneously.

Here is our software folder tree structure:
<img width="599" alt="image" src="https://user-images.githubusercontent.com/90274880/208264520-3fd71a63-ac37-49bd-910a-5356384b453c.png">

The requirements our control software is trying to achieve can be broken down into two different categories: the driving requirements and the license plate detection requirements. 

The driving requirements are the following:
The robot must drive around of the course while avoiding collisions with pedestrians at crosswalks.
The robot must have at least 2 wheels within the white lines of the road at all times.
The robot must complete a lap of the outer loop in less than 4 minutes.

The license plate detection requirements are the following:
The robot must be able to detect every blue parked car along the outer loop.
The robot must be able to return license plate numbers while tracking the associated parking number of each parked car.

