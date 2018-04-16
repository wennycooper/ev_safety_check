#!/usr/bin/env python
from __future__ import print_function

from threading import Timer
import sys
import rospy
import rospkg
import numpy as np
import cv2
from std_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import load_model


rospack = rospkg.RosPack()
packPath = rospack.get_path('ev_safety_check')
print(packPath)

bridge = CvBridge()
count = 0
# load
modelFileName = packPath + "/models/my_model.h5"
model = load_model(modelFileName)
result = model.predict(np.zeros((1,48,64,3)))  # this line makes it not crashed

checkCBPub = rospy.Publisher('/checkEVcb',Bool,queue_size=1)

startCheck = False
safeCount = 0
unSafeCount = 0

# testList = [0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1]

def callback(data):
    global safeCount, unSafeCount, count, startCheck

    # if the startCheck flag is not True, return immediately
    if not startCheck:
        return

    count = count + 1

    # sampling
    if count != 2:
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
        # print(prediction[0])
        if prediction[0] == 1:
            safeCount = safeCount + 1
        elif prediction[0] == 0:
            unSafeCount = unSafeCount + 1

    '''
    # test purpose
    print(testList)
    pred = testList.pop(0)
    if pred == 1:
        safeCount = safeCount + 1
    elif pred == 0:
        unSafeCount = unSafeCount + 1
    '''

def chekc_elevator(msg):
    global safeCount, unSafeCount, startCheck, count
    safeCount = 0
    unSafeCount = 0
    count = 0
    startCheck = True

    t = Timer(3 ,chekc_elevatorCB)
    t.daemon = True
    t.start()
    return

def chekc_elevatorCB():
    global saftCount, unSafeCount, startCheck, checkCBPub
    rospy.loginfo('[ev_safety_check] result(safe, unSafe): ' + str(safeCount) + ', ' + str(unSafeCount))
    if safeCount >= 12:
        checkCBPub.publish(True)
    else:
        checkCBPub.publish(False)

    # reset the startCheck flag
    startCheck = False
    
    return 

        

def main(args):
    rospy.init_node('ev_safty_check_test', anonymous=True)
    image_sub = rospy.Subscriber("/camera_rear/image_rect_color", Image, callback)

    rospy.Subscriber('/checkEV',Bool,chekc_elevator)
    now = rospy.Time.now()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
