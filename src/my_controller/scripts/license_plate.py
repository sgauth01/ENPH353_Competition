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
        image_wanted, h, w, rb, rt, lb, lt, ratio_a = self.four_corner_points(rgb_frame, contours)
        if h > 50 and ratio_a:
          #cv2.imshow('Wanted', image_wanted)
          #cv2.waitKey(1)

          sepimage = self.separatingimages(image_wanted)

          transform = self.perspectiveTransform(image_wanted, lt, rb, lb, rt)

          cv2.imshow("transform", transform)
          cv2.waitKey(1)

        
          bool, newcontours = (self.twoLargestContours(sepimage))

          #print(newcontours[1])
    
          if (bool == False):
            parkingimage, rb_pa, rt_pa, lb_pa, lt_pa = self.four_corner_points_parking(sepimage, newcontours)


            licenseplateimage, rb_lp, rt_lp, lb_lp, lt_lp = self.four_corner_point_license_plate(image_wanted, newcontours)



    # find width and height of image
    height = gray_frame.shape[0]
    width = gray_frame.shape[1]

  def findingcar(self, hsv):
    counter = 0
    foundCar = False

    low_blue = np.array([104, 173, 69])
    high_blue = np.array([154, 255, 255])
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

    try:
        ratio = height/width
    except ZeroDivisionError:
        ratio = 0

    ratio_acceptable = False
    
    if ratio>1 and ratio<1.5:
        ratio_acceptable = True


    return new_image, height, width, right_bot, right_top, left_bot, left_top, ratio_acceptable

  def separatingimages(self, image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    gray_mask = cv2.inRange(image_hsv, (0,0,90), (0,0,210)) #white pixels in range and black pixels not in range
    gray = cv2.bitwise_and(image_hsv, image_hsv, mask=gray_mask) 

    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_HSV2RGB)
    grayscale = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2GRAY)
    _, gray_binary = cv2.threshold(grayscale, 10, 255, cv2.THRESH_BINARY)

    closed_bin = self.closing(gray_binary)

    return closed_bin 

  def four_corner_points_parking(self, image, contours):
    largest_contour = contours[0]
    print(largest_contour)
    rect_largest = cv2.minAreaRect(largest_contour)
    box_largest = cv2.boxPoints(rect_largest)
    box_largest = np.int0(box_largest)

    minx_1 = np.min(box_largest[:,0])
    miny_1 = np.min(box_largest[:,1])

    maxx_1 = np.max(box_largest[:,0])
    maxy_1 = np.max(box_largest[:,1])

    right_bot = (maxx_1, maxy_1)
    right_top = (maxx_1, miny_1)
    left_bot = (minx_1, maxy_1)
    left_top = (minx_1, miny_1)

    cv2.circle(image, right_bot, 8, (255, 255, 0), -1) # gives bottom left
    cv2.circle(image, right_top, 8, (255, 255, 0), -1) # gives bottom left
    cv2.circle(image, left_bot, 8, (255, 255, 0), -1) # gives bottom left
    cv2.circle(image, left_top, 8, (255, 255, 0), -1) # gives bottom left

    new_image = image[left_top[1]:left_bot[1],left_bot[0]:right_bot[0]]

  #plt.imshow(new_image)

    return new_image, right_bot, right_top, left_bot, left_top


  def four_corner_point_license_plate(self, image, contours):
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


    right_bot = (maxx_2, miny_2)
    right_top = (maxx_2, maxy_1)
    left_bot = (minx_2, miny_2)
    left_top = (minx_2, maxy_1)
  
    cv2.circle(image, right_bot, 8, (255, 255, 0), -1) # gives bottom left
    cv2.circle(image, right_top, 8, (255, 255, 0), -1) # gives bottom left
    cv2.circle(image, left_bot, 8, (255, 255, 0), -1) # gives bottom left
    cv2.circle(image, left_top, 8, (255, 255, 0), -1) # gives bottom left

    new_image = image[left_top[1]:left_bot[1],left_bot[0]:right_bot[0]]

    return new_image, right_bot, right_top, left_bot, left_top

  def perspectiveTransform(self, image, lt, rb, lb, rt):
    pts1 = np.float32([lt, rb, lb, rt])
    pts2 = np.float32([[0,0], [180,232], [0,232], [180,0]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (180,232))

    return dst
    #plt.imshow(dst)

#todo:
# pers transform-> crop the LP and parking number

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
