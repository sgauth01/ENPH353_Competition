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

class image_converter:

  def __init__(self):
    self.command_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)

    self.plate_pub = rospy.Publisher("/license_plate", String, queue_size=1)

    self.start_time = time.time()
    self.number_of_plates_tracked = 0
    self.detected_time = 0
    self.pedestrian_detected = False
    self.state = 0
    self.crosswalk_init_time = 0
    self.num_crosswalks_detected = 0
    self.last_crosswalk_time = 0
    self.last_plate_published = False

    # decide which line of the road to follow (0 is left and 1 is right)
    self.line_to_follow = 0

    # self.plate_pub = rospy.Publisher("/license_plate", String, queue_size=1)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # convert image to grayscale
    gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # find width and height of image
    height = gray_frame.shape[0]
    width = gray_frame.shape[1]

    move = Twist()
    
    # convert image to hsv image
    hsv_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # convert hsv image to binary scale
    low_white = np.array([0, 0, 100])
    high_white = np.array([0, 0, 255])
    binary_hsv = cv2.inRange(hsv_frame, low_white, high_white)
    
    # blur image
    binary_img = cv2.blur(binary_hsv,(5,5))

    if (self.state == 0):

      # STATE: DRIVING AROUND OUTER LOOP

      self.pedestrian_detected = False

      current_time = time.time()

      if (current_time - self.start_time > 6):
        self.line_to_follow = 1

      row = binary_img[600]

      kp = 4

      leading_edge = 0
      origin = width/2
      move.linear.x = 0.2
      threshold = 200
      sum = 0

      # detect the edge of the line on the left side
      if (self.line_to_follow == 0):
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

      # TRYING TO DETECT CROSSWALK:

      # filter out every colour except red
      low_red = np.array([0, 200, 0])
      high_red = np.array([6, 255, 255])
      red_mask = cv2.inRange(hsv_frame, low_red, high_red)
      red_blurry = cv2.blur(red_mask,(5,5))

      M = cv2.moments(red_blurry)
      try:
        cY = int(M["m01"] / M["m00"])
      except ZeroDivisionError:
        cY = 0
      
      current_time = time.time()

      if (cY > 500) and (current_time - self.detected_time > 5):
        self.state = 1 # crosswalk state
        self.crosswalk_init_time = time.time()
      
      # FINDING TIME ATER TWO CROSSWALKS WERE DETECTED
      if (self.num_crosswalks_detected == 2):
        self.last_crosswalk_time = time.time()
        self.num_crosswalks_detected += 1
      
      # STOPPING SIMULATION AFTER A FEW SECONDS OF TWO CROSSWALKS BEING DETECTED
      if (current_time - self.last_crosswalk_time > 12) and (self.num_crosswalks_detected == 3):
        # stop moving
        move.linear.x = 0
        move.angular.z = 0
        center = 0
        if (self.last_plate_published == False):
          self.plate_pub.publish(finalPlate)
          self.last_plate_published = True
      
    # STATE: CROSSWALK
    elif (self.state == 1):
      # stop moving
      move.linear.x = 0
      move.angular.z = 0
      center = 0

      cropped = binary_img[360:380,650:670]

      # try to detect pedestrian
      average_colour = np.mean(cropped)

      if average_colour > 5:
        self.pedestrian_detected = True
        print("Pedestrian Detected")
        self.state = 0
        self.num_crosswalks_detected += 1
        self.detected_time = time.time()
      
      current_time = time.time()
      if ((current_time - self.crosswalk_init_time) > 8):
        self.detected_time = time.time()
        self.num_crosswalks_detected += 1
        self.state = 0

    # highlight the center of the road

    cv2.circle(cv_image, (center, 600), 5, (255, 255, 255), -1)
    cv2.putText(cv_image, "center", (center - 25, 600 - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # write at top left of image which line is being followed

    if (self.line_to_follow == 0):
      cv2.putText(cv_image, "line following: left", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
      cv2.putText(cv_image, "line following: right", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if (self.state == 0):
      cv2.putText(cv_image, "state: driving", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    elif (self.state == 1):
      cv2.putText(cv_image, "state: crosswalk", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if (self.pedestrian_detected):
      cv2.putText(cv_image, "PEDESTRIAN DETECTED!!!!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # display the car mask

    # display the normal CV image
    cv2.imshow('CV Image', cv_image)
    cv2.waitKey(3)

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
