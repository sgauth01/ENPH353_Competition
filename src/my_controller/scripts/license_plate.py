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

class image_converter:

  def __init__(self):
    self.command_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback),

    self.plate_pub = rospy.Publisher("/license_plate", String, queue_size=1)

    self.car_detected = False
    self.number_of_plates_tracked = 0
    self.count_parking = 0

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

    rgb_frame_copy = rgb_frame.copy()

    largest_contour, area, found_plate = self.findContour(rgb_frame_copy)

    if (found_plate == True) and (area > 10000):
      new_rgb = rgb_frame.copy()
      corners, area = self.four_corner_points(new_rgb, largest_contour)
      if (area != 0):
        full, plate = self.perspectiveTransform(rgb_frame, corners)
        #count_parking = count_parking+1
        #parking_number_value = count_parking
        let1plate, let2plate, let3plate, let4plate = self.hsv_filter_plate(plate)
        #cv2.imshow("transform", full)
        #cv2.waitKey(3)
        #cv2.imshow("plate only", plate)
        cv2.imshow("let1plate", let1plate)
        cv2.waitKey(3)

  def findContour(self, image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    found_plate = False

    lower = np.array([0,0,90])
    upper = np.array([0,0,210])
    mask = cv2.inRange(hsv_image,lower,upper)
    count = np.sum(mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    largest_contour = contours_sorted[0]

    # get the area of the contour
    area = cv2.contourArea(largest_contour)

    contour_color = (0, 255, 0)

    withcont = cv2.drawContours(image, [largest_contour], 0, contour_color, 5)
    cv2.imshow("With Contours", withcont)

    if count > 0:
      found_plate = True

    return largest_contour, area, found_plate

  def order_points(self, pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

  def four_corner_points(self, image, contour):

    epsilon = 0.05*cv2.arcLength(contour,True)
    box = cv2.approxPolyDP(contour, epsilon, True)

    # see if there are 4 corners
    if box.shape[0] != 4:
        return (None, 0)
    
    # if there are 4 corners then sort them
    box2 = np.zeros((4,2))
    for i in range(4):
        box2[i,:] = box[i,0,:]
    corners = self.order_points(box2)

    # get the are of the contour
    area = cv2.contourArea(contour)

    return corners, area

  def perspectiveTransform(self, image, corners):
    pts1 = np.float32(np.array(corners))
    pts2 = np.float32(np.array([(0, 0), (300,0), (300, 300), (0,300)]))

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_with_plate = cv2.warpPerspective(image, matrix, (300,380))
    only_plate = transformed_with_plate[300:380,:]
    large_plate = cv2.resize(only_plate, (600, 298), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Transformed Full", transformed_with_plate)

    self.count_parking = self.count_parking+1
    print(self.count_parking) 

    return transformed_with_plate, large_plate

  def hsv_filter_plate(self, image): 
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([117,113,82])
    upper = np.array([255,255,255])
    mask = cv2.inRange(hsv_image,lower,upper) 

    first_letter = image[80:250,40:150]
    second_letter = image[80:250,150:260]
    third_letter = image[80:250,340:450]
    fourth_letter = image[80:250,450:560]

    return first_letter, second_letter, third_letter, fourth_letter

    # todo: cropping, counting parking

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv) 
