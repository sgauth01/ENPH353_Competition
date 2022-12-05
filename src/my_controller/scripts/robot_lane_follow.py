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

class image_converter:

  number_of_plates_tracked = 0
  detected_time = 0 # a large value
  pedestrian_detected = False

  def __init__(self):
    self.command_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback),

    self.plate_pub = rospy.Publisher("/license_plate", String, queue_size=1)

    self.car_detected = False

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    state = 0 # driving around outer loop
    self.pedestrian_detected = False

    # convert image to grayscale
    gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # convert image to hsv image
    hsv_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # convert image to rgb image
    rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    self.car_detected, car_mask = self.findingcar(hsv_frame)

    if (self.car_detected):
      closed_car_mask = self.closing(car_mask)
      closed_car_mask_copy = closed_car_mask.copy()
      no_plates, contours = self.twoLargestContours(closed_car_mask_copy)
      if (no_plates == False):
        image_wanted, h, w, rb, rt, lb, lt = self.four_corner_points(rgb_frame, contours)
        if h > 50 and w > 50:
          cv2.imshow('Wanted', image_wanted)
          cv2.waitKey(1)

    # find width and height of image
    height = gray_frame.shape[0]
    width = gray_frame.shape[1]


    if (state == 0):

      # STATE: DRIVING AROUND OUTER LOOP

      # after number of plates tracked = 1, switch to following right line (from left line)
      # for now let's just say that when 5 seconds have passed we switch the line

      # decide which line of the road to follow (0 is left and 1 is right)
      line_to_follow = 0

      current_time = time.time()

      if (current_time - start_time > 6):
        line_to_follow = 1

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
    
    # filter out every colour except red
    red_mask = cv2.inRange(hsv_frame, (0, 200, 0), (6, 255, 255))
    red_blurry = cv2.blur(red_mask,(5,5))

    M = cv2.moments(red_blurry)
    # try:
    #   cY = int(M["m01"] / M["m00"])
    # except ZeroDivisionError:
    #   cY = 0
    
    # current_time = time.time()

    # if cY > 500:
    #   state = 1 #crosswalk state
      
    # STATE: CROSSWALK
    if (state == 1):
      # stop moving
      move.linear.x = 0
      move.angular.z = 0

      # track time at start of crosswalk
      crosswalk_init_time = time.time()

      # try to detect pedestrian
      pixel_row = binary_img[375][650:670]
      for pixel in pixel_row:
        if (pixel > 200):
          self.pedestrian_detected = True
          state = 0
          self.detected_time = time.time()
          break
      
      current_time = time.time()
      if ((current_time - crosswalk_init_time) > 10):
        self.detected_time = time.time()
        state = 0

    # highlight the center of the road

    cv2.circle(cv_image, (center, 600), 5, (255, 255, 255), -1)
    cv2.putText(cv_image, "center", (center - 25, 600 - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # write at top left of image which line is being followed

    # if (line_to_follow == 0):
    #   cv2.putText(cv_image, "line following: left", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # else:
    #   cv2.putText(cv_image, "line following: right", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # if (state == 0):
    #   cv2.putText(cv_image, "state: driving", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # elif (state == 1):
    #   cv2.putText(cv_image, "state: crosswalk", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # if (self.pedestrian_detected):
    #   cv2.putText(cv_image, "PEDESTRIAN DETECTED!!!!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # # display the binary HSV image
    # cv2.imshow('Binary HSV', binary_img)
    # cv2.waitKey(1)

    # display the car mask
    cv2.imshow('Car Mask', car_mask)
    cv2.waitKey(1)

    # display the normal CV image
    cv2.imshow('CV Image', cv_image)
    cv2.waitKey(1)

    # after 30 seconds have passed, stop timer
    current_time = time.time()
    if (current_time - start_time) > 30 and (current_time - start_time) < 31:
      self.plate_pub.publish(finalPlate)

    try:
      self.command_pub.publish(move)
    except CvBridgeError as e:
      print(e)

  def findingcar(self, hsv):
    counter = 0
    foundCar = False

    low_blue = np.array([119, 50, 50])
    high_blue = np.array([124, 255, 255])
    blue_mask = cv2.inRange(hsv, low_blue, high_blue) #white pixels in range and black pixels not in range
    blue = cv2.bitwise_and(hsv, hsv, mask=blue_mask) 

    #code for displaying the gray part
    indices = np.nonzero(blue_mask)
    size = len(set(zip(indices[0], indices[1])))
    avgX, avgY = 0, 0
    minX, minY = 10000, 10000
    maxX, maxY = -1, -1

    for (x, y) in set(zip(indices[0], indices[1])):
        minX, minY, maxX, maxY = min(minX, x), min(minY, y), max(maxX, x), max(maxY, y)

    rectangle = cv2.rectangle(blue_mask, (minX, minY), (maxX, maxY), (0,0,0), -1)

    gray_rgb = cv2.cvtColor(blue, cv2.COLOR_HSV2RGB)
    grayscale = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2GRAY)
    _, gray_binary = cv2.threshold(grayscale, 10, 255, cv2.THRESH_BINARY)

    count = np.sum(blue_mask)
    
    if count>0:
        foundCar = True
        counter += 1

    return foundCar, gray_binary

  def closing(self, carmask):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  
    # opening the image
    closed_image = cv2.morphologyEx(carmask, cv2.MORPH_CLOSE, kernel, iterations=1)
  
    # print the output

    return closed_image

  def twoLargestContours(self, image):

    no_plates = False

    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    twoContours = []
    areas = []
    dimensions = []
    twoareas = []
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)

    if (len(contours) <= 1):
      no_plates = True

    else:
      ordered_contours = sorted(contours, key=cv2.contourArea, reverse=True)
      areas = [cv2.contourArea(c) for c in ordered_contours]

      c = max(contours, key = cv2.contourArea)

      largestcontour = ordered_contours[0]

      contour_color = (0, 255, 0)
      contour_thick = 10
      x, y, w, h = cv2.boundingRect(largestcontour)
      dimensions.append(x)
      dimensions.append(y)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
      withcont1 = cv2.drawContours(image_rgb, [largestcontour], 0, contour_color, 10)

      secondlargestcontour = ordered_contours[1]

      x_two, y_two, w_two, h_two = cv2.boundingRect(secondlargestcontour)
      dimensions.append(x_two)
      dimensions.append(y_two)
      withcont2 = cv2.drawContours(image_rgb, [secondlargestcontour], 0, contour_color, 10)

      twoContours = [largestcontour, secondlargestcontour]

    return no_plates, twoContours
  
  def four_corner_points(self, image, contours):
    largest_contour = contours[0]
    second_largest_contour = contours[1]

    rect_largest = cv2.minAreaRect(largest_contour)
    box_largest = cv2.boxPoints(rect_largest)
    box_largest = np.int0(box_largest)

    minx_1 = np.min(box_largest[:,0])
    miny_1 = np.min(box_largest[:,1])

    maxx_1 = np.max(box_largest[:,0])
    maxy_1 = np.max(box_largest[:,1])

    rect_second_largest = cv2.minAreaRect(second_largest_contour)
    box_second_largest = cv2.boxPoints(rect_second_largest)
    box_second_largest = np.int0(box_second_largest)

    minx_2 = np.min(box_second_largest[:,0])
    miny_2 = np.min(box_second_largest[:,1])

    maxx_2 = np.max(box_second_largest[:,0])
    maxy_2 = np.max(box_second_largest[:,1])

    # if largest to the right
    if maxx_1 > maxx_2:
      right_bot = (minx_1, maxy_1)
      right_top = (minx_1, miny_1)
      left_bot = (maxx_2, maxy_2)
      left_top = (maxx_2, miny_2)

    # if largest to the left
    else:
      right_bot = (minx_2, maxy_2)
      right_top = (minx_2, miny_2)
      left_bot = (maxx_1, maxy_1)
      left_top = (maxx_1, miny_1)

    cv2.circle(image, right_bot, 8, (255, 255, 0), -1) # gives bottom left
    cv2.circle(image, right_top, 8, (255, 255, 0), -1) # gives bottom left
    cv2.circle(image, left_bot, 8, (255, 255, 0), -1) # gives bottom left
    cv2.circle(image, left_top, 8, (255, 255, 0), -1) # gives bottom left

    new_image = image[left_top[1]:left_bot[1],left_bot[0]:right_bot[0]]
    
    height = new_image.shape[0]
    width = new_image.shape[1]

    return new_image, height, width, right_bot, right_top, left_bot, left_top

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
