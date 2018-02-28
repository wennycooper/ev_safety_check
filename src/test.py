#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import numpy as np
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import load_model



bridge = CvBridge()
count = 0
# load
modelFileName = "/home/kkuei/catkin_ws/src/ev_safety_check/models/my_model.h5"
model = load_model(modelFileName)
result = model.predict(np.zeros((1,48,64,3)))  # this line makes it not crashed

def callback(data):
    global count
    count = count + 1
    if count != 30:
        return
    count = 0

    try:
        cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
        cv2_img = cv2.resize(cv2_img, (64, 48))
        # resize, grey out ...etc
    except CvBridgeError as e:
        print(e)
    else:
        #fname = "/home/kkuei/catkin_ws/src/ev_safety_check/test/safe/1519704213556588888_64x48.jpeg"
        #cv2_img = cv2.imread(fname)
        np_img = np.reshape(cv2_img, (1,48,64,3)).astype('float32')
        np_img_normalized = np_img/255
        
        prediction = model.predict_classes(np_img_normalized, verbose=0)
        # 1: safe, 0: unsafe
        print(prediction[0])
        

def main(args):
    rospy.init_node('ev_safty_check_test', anonymous=True)
    image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, callback)

    now = rospy.Time.now()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
