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

    if (current_time - start_time > 10):
      line_to_follow = 1
    
    # convert image to grayscale
    gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # find width and height of image
    height = gray_frame.shape[0]
    width = gray_frame.shape[1]

    # convert image to binary scale
    (thresh, binary_img) = cv2.threshold(gray_frame, 180, 255, cv2.THRESH_BINARY)
    
    # blur image
    binary_img = cv2.blur(binary_img,(5,5))

    # crop image to specific location
    cropped_ahead = binary_img[400:500,:]
    cropped_nav = binary_img[600:700,:]

    # calculate moments of navigation cropped binary image
    M_n = cv2.moments(cropped_nav)
    
    # calculate x,y coordinate of center
    if M_n["m00"] != 0:
      cX_n = int(M_n["m10"] / M_n["m00"]) 
      cY_n = int(M_n["m01"] / M_n["m00"]) + 600
    else:
      cX_n, cY_n = 0, 0
    
    # calculate moments of ahead cropped binary image
    M_a = cv2.moments(cropped_ahead)
    
    # calculate x,y coordinate of center
    if M_a["m00"] != 0:
      cX_a = int(M_a["m10"] / M_a["m00"])
      cY_a = int(M_a["m01"] / M_a["m00"]) + 400
    else:
      cX_a, cY_a = 0, 0

    # if (cX_a < cX_n) and (cY_a < cX_a):
    #   line_to_follow = 1
    
    # line follows based on the line to follow

    row = binary_img[600]

    # Goes through each pixel's gray value in bottom_row and find the column number of the first 
    # pixel that has a gray value below the threshold.
    # Note: darker pixels have smaller values than lighter pixels

    move = Twist()
    kp = 4

    leading_edge = 0
    origin = width/2
    move.linear.x = 0.2
    threshold = 150
    sum = 0

    if (line_to_follow == 0):
      column = 0
      for pixel in row:
        column += 1
        if (pixel > threshold):
          sum += 1
        else:
          sum = 0
        if (sum >= 30):
          leading_edge = column - sum
          break

      center = leading_edge + 400
      move.angular.z = (1-center/origin)*kp
    
    else:
      for i in range(1, len(row)):
        column = width - i
        if (row[column] > threshold):
          sum += 1
        else:
          sum = 0
        if (sum >= 30):
          leading_edge = column + sum
          break

      center = leading_edge - 400 
      move.angular.z = (1-center/origin)*kp 


    # else:
    #   move.linear.x = 0.1
    #   move.angular.z = 0
    #   center = int(width/2)

    # put text and highlight the centroid

    cv2.circle(binary_img, (leading_edge, 600), 5, (255, 255, 255), -1)
    cv2.putText(binary_img, "leading edge", (leading_edge - 25, 600 - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.circle(binary_img, (cX_n, cY_n), 5, (255, 255, 255), -1)
    cv2.putText(binary_img, "nav centroid", (cX_n - 25, cY_n - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.circle(binary_img, (cX_a, cY_a), 5, (255, 255, 255), -1)
    cv2.putText(binary_img, "ahead centroid", (cX_a - 25, cY_a - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.circle(binary_img, (center, 600), 5, (255, 255, 255), -1)
    cv2.putText(binary_img, "center", (center - 25, 600 - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if (line_to_follow == 0):
      cv2.putText(binary_img, "line following: left", (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
      cv2.putText(binary_img, "line following: right", (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Black white image', binary_img)
    cv2.waitKey(3)
 
    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)

    current_time = time.time()
    if (current_time - start_time) > 5 and (current_time - start_time) < 6:
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