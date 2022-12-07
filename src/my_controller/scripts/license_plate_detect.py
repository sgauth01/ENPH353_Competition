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
from tensorflow.keras.models import Sequential, model_from_json

class image_converter:

  def __init__(self):
    self.command_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback),

    self.plate_pub = rospy.Publisher("/license_plate", String, queue_size=1)

    self.car_detected = False
    self.number_of_plates_tracked = 0

    self.json_file = open('model.json', 'r')
    self.loaded_model_json = self.json_file.read()
    self.json_file.close()

    self.loaded_model = model_from_json(self.loaded_model_json)
    # load weights into new model
    self.loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    self.plate_init_time = 0

    # initialize to regular driving state
    self.state = 0

    self.first_letter = []
    self.second_letter = []
    self.third_letter = []
    self.fourth_letter = []

    self.first = ''
    self.second = ''
    self.third = ''
    self.fourth = ''

    self.entire_plate = ''

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # plate = cv2.imread('/home/fizzer/Desktop/plate_2.png')
    # cv2.imshow("plate", plate)
    # cv2.waitKey(3)

    # convert image to grayscale
    gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # convert image to hsv image
    hsv_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # convert image to rgb image
    rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    rgb_frame_copy = rgb_frame.copy()

    # get the largest contour and area
    largest_contour, area, found_plate = self.findContour(rgb_frame_copy)

    # proceed with next steps if a contour was found and the area of the contour is greater than 10000
    if (found_plate == True) and (area > 10000):

      if (self.state == 0):
        self.plate_init_time = time.time()
        # change state to plate detecting state if was in driving state before
        self.state = 1

      new_rgb = rgb_frame.copy()

      # find the corners and area of the largest contour
      corners, area = self.fourCornerPoints(new_rgb, largest_contour)

      # if the area is not 0 (and there is a contour present), take a perspective transform of the contour
      if (area != 0):
        full, plate = self.perspectiveTransform(cv_image, corners)
        cv2.imshow("transform", full)
        cv2.waitKey(3)
        cv2.imshow("plate only", plate)
        cv2.waitKey(3)
        processed = self.processImage(plate)
        cv2.imshow("processed", processed)
        first, second, third, fourth = self.splitLetters(processed)

        if (self.state == 1):
          self.first_letter.append(self.getPrediction(first))
          self.second_letter.append(self.getPrediction(second))
          self.third_letter.append(self.getPrediction(third))
          self.fourth_letter.append(self.getPrediction(fourth))
          
    current_time = time.time()
    
    # after two seconds, exit plate detecting state
    if (current_time - self.plate_init_time > 2) and (self.state == 1):

      # get most frequent letter from the list of letters
      if (len(self.first_letter) > 0):
        self.first = self.getMostFrequent(self.first_letter)
        self.entire_plate += self.first
        self.second = self.getMostFrequent(self.second_letter)
        self.entire_plate += self.second
        self.third = self.getMostFrequent(self.third_letter)
        self.entire_plate += self.third
        self.fourth = self.getMostFrequent(self.fourth_letter)
        self.entire_plate += self.fourth
        print(self.entire_plate)
      else:
        self.first = 0
        self.second = 0
        self.third = 0
        self.fourth = 0

      # reinitialize the list
      self.first_letter = []
      self.second_letter = []
      self.third_letter = []
      self.fourth_letter = []

      # reintialize entire plate prediction
      self.entire_plate = ''

      # go back to driving state
      self.state = 0

  def findContour(self, image):

    found_plate = False

    # convert image to hsv scale
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # range of hsv values for the white-grayish colour of the parking number plate
    lower = np.array([0,0,90])
    upper = np.array([0,0,210])
    mask = cv2.inRange(hsv_image,lower,upper)

    # count how many contours were find
    count = np.sum(mask)

    if count > 0:
      found_plate = True

    # find all contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort by contour by area
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    # get contour with largest area
    largest_contour = contours_sorted[0]

    # get the area of the largest contour
    area = cv2.contourArea(largest_contour)

    contour_color = (0, 255, 0)

    # draw contours on image
    withcont = cv2.drawContours(image, [largest_contour], 0, contour_color, 5)
    cv2.imshow("With Contours", withcont)

    return largest_contour, area, found_plate

  def orderPoints(self, points):

    rectangle = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)

    rectangle[0] = points[np.argmin(s)]
    rectangle[2] = points[np.argmax(s)]

    diff = np.diff(points, axis = 1)
    rectangle[1] = points[np.argmin(diff)]
    rectangle[3] = points[np.argmax(diff)]

    return rectangle

  def fourCornerPoints(self, image, contour):

    # find the box around the contour
    epsilon = 0.05*cv2.arcLength(contour,True)
    box = cv2.approxPolyDP(contour, epsilon, True)

    num_corners = box.shape[0]

    # check that the box has 4 corners
    # if not return no corners, and an area of 0
    if num_corners != 4:
        return (None, 0)
    
    # get all four corner points
    rectangle = np.zeros((4,2))
    for i in range(0, 4):
        rectangle[i,:] = box[i,0,:]

        # order corner points
    corners = self.orderPoints(rectangle)

    # get contour area
    area = cv2.contourArea(contour)

    return corners, area

  def perspectiveTransform(self, image, corners):
    pts1 = np.float32(np.array(corners))
    pts2 = np.float32(np.array([(0, 0), (300,0), (300, 300), (0,300)]))

    # get perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # extend frame a little to also get license plate
    transformed_with_plate = cv2.warpPerspective(image, matrix, (300,380))

    # crop to only get the plate
    only_plate = transformed_with_plate[300:380,:]

    # resize plate image
    large_plate = cv2.resize(only_plate, (600, 298), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Transformed Full", transformed_with_plate)

    return transformed_with_plate, large_plate
  
  def processImage(self, plate):
    # gray scale the plate
    plate_bw = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # apply binary filter on plate
    th,img = cv2.threshold(plate_bw,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return img
  
  def splitLetters(self, plate):
    first_letter = plate[80:250,40:150]
    second_letter = plate[80:250,150:260]
    third_letter = plate[80:250,340:450]
    fourth_letter = plate[80:250,450:560]

    return first_letter, second_letter, third_letter, fourth_letter
  
  def getPrediction(self, letter):
    new_y_pred = ""

    # use model to make character prediction based on input image
    img_aug = np.expand_dims(letter, axis=0)
    y_predict = self.loaded_model.predict(img_aug)[0]

    # get make value from y_predict and convert it to its associate character
    predictions = y_predict.tolist()
    max_value = max(predictions)
    max_index = predictions.index(max_value)
    new_y_pred += self.numEncoding(max_index)

    return new_y_pred
  
  def getMostFrequent(self, input_list):
    # finds most frequent element in a list
    return max(set(input_list), key = input_list.count)
  
  def numEncoding(self, number):
  # matches each number to a letter
    number_encoding = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'J',
        10: 'K',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T',
        20: 'U',
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: 'Z',
        26: '0',
        27: '1',
        28: '2',
        29: '3',
        30: '4',
        31: '5',
        32: '6',
        33: '7',
        34: '8',
        35: '9',
    }

    return number_encoding[number]

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