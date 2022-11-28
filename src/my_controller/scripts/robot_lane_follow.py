#! /usr/bin/env python3

from __future__ import print_function
from geometry_msgs.msg import Twist
import sys
import rospy
import cv2
from std_msgs.msg import String
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

initPlate = 'TeamTS,bear,0,XR58'
finalPlate = 'TeamTS,bear,-1,XR58'
start_time = time.time()

number_of_plates_tracked = 0

class image_converter:

  def __init__(self):
    self.command_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback),

    self.plate_pub = rospy.Publisher("/license_plate", String, queue_size=1)
    
  
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # after number of plates tracked = 1, switch to following right line (from left line)
    # for now let's just say that when 5 seconds have passed we switch the line

    # decide which line of the road to follow (0 is left and 1 is right)
    line_to_follow = 0

    current_time = time.time()

    if (current_time - start_time > 8):
      line_to_follow = 1
    
    # convert image to grayscale
    gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # convert image to hsv image
    hsv_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # find width and height of image
    height = gray_frame.shape[0]
    width = gray_frame.shape[1]

    # convert grayscale image to binary scale
    (thresh, binary_img) = cv2.threshold(gray_frame, 180, 255, cv2.THRESH_BINARY)

    # convert hsv image to binary scale
    binary_hsv = cv2.inRange(hsv_frame, (0, 0, 100), (0, 0, 255))
    
    # blur image
    binary_img = cv2.blur(binary_hsv,(5,5))

    row = binary_img[600]

    move = Twist()
    kp = 4

    leading_edge = 0
    origin = width/2
    move.linear.x = 0.2
    threshold = 200
    sum = 0

    # detect the edge of the line on the left side
    if (line_to_follow == 0):
      column = 0
      for pixel in row:
        column += 1
        if (pixel > threshold):
          # making sure there are at least 5 white pixels in a row and not just random noisy pixels
          sum += 1
        else:
          sum = 0
        if (sum >= 5):
          leading_edge = column - sum
          break

      # find the center pixel
      center = leading_edge + 400
      #find how much turning is needed
      move.angular.z = (1-center/origin)*kp
    
    # detect the edge of the line on the right side
    else:
      for i in range(1, len(row)):
        column = width - i
        if (row[column] > threshold):
          sum += 1
        else:
          sum = 0
        if (sum >= 5):
          leading_edge = column + sum
          break
      
      # find the center pixel
      center = leading_edge - 400 
      # find how much turning is needed
      move.angular.z = (1-center/origin)*kp 

    # highlight the center of the road

    cv2.circle(binary_img, (center, 600), 5, (255, 255, 255), -1)
    cv2.putText(binary_img, "center", (center - 25, 600 - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # write at top left of image which line is being followed

    if (line_to_follow == 0):
      cv2.putText(binary_img, "line following: left", (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
      cv2.putText(binary_img, "line following: right", (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # display the binary HSV image
    cv2.imshow('Binary HSV', binary_img)
    cv2.waitKey(3)

    # after 30 seconds have passed, stop timer
    current_time = time.time()
    if (current_time - start_time) > 30 and (current_time - start_time) < 31:
      self.plate_pub.publish(finalPlate)

    try:
      self.command_pub.publish(move)
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)

  ic.plate_pub.publish(initPlate)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
