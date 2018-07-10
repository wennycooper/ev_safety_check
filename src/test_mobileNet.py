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
#from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras_applications.mobilenet import relu6
from keras.layers import convolutional
from keras.applications.imagenet_utils import decode_predictions
import time


continuousSafetyCheckLPFGain = 0.9
continuousSafetyCheckScore = 1.0
continuousSafetyCheckScoreThreshold = 0.9

rospack = rospkg.RosPack()
packPath = rospack.get_path('ev_safety_check')
print(packPath)

bridge = CvBridge()
count = 0
# load
modelFileName = packPath + "/models/mobileNet_ex1.h5"
model = load_model(modelFileName,custom_objects={
                   'relu6': relu6,
                   'DepthwiseConv2D': convolutional.DepthwiseConv2D})

result = model.predict(np.zeros((1,224,224,3)))  # this line makes it not crashed

checkCBPub = rospy.Publisher('/checkEVcb',Bool,queue_size=1)
continuousSafetyCheckResultUnsafePub = rospy.Publisher('/continuousSafetyCheckResultUnsafe',Bool,queue_size=1)

startCheck = False
safeCount = 0
unSafeCount = 0

chekc_elevatorFlag = False
continuousT1_pass = False
continuousSafetyCheckStartFlag = False
totalTime = 0

# testList = [0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1]

def imagePrediction(data):
    global safeCount, unSafeCount, count, startCheck, totalTime, continuousSafetyCheckStartFlag
    #print("==[callback]==")
    # if the startCheck flag is not True, return immediately
    if not startCheck:
        return
        
    
    start1  = time.time()
    count = count + 1
    #print("[callback] count :",count ,"time :", time.asctime (time.localtime(start1) ))
    #print("[callback] data =", data)
    #print("[callback] data size =", sys.getsizeof(data))

    # sampling
    #if count != 2:
    #    return
    #count = 0

    try:
        cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
        #cv2.imshow("Image window", cv2_img)
        #cv2.waitKey(1)
        cv2_img = cv2.resize(cv2_img, (224, 224))
        
        # resize, grey out ...etc
    except CvBridgeError as e:
        print(e)
    else:
    
        np_img = np.reshape(cv2_img, (1,224,224,3)).astype('float32')
        np_img_normalized = np_img/255
        
        #end1 = time.time()

        #start2 = time.time()

        #prediction = model.predict_classes(np_img_normalized, verbose=0)
        prediction = model.predict(np_img_normalized, verbose=0)
        label = prediction.argmax(axis=-1)
        #print ("prediction = ",prediction)
        #print ('result = ', label[0])
        
        end2 = time.time()

        #0: safe, 1: unsafe
        if label[0] == 0:
            safeCount = safeCount + 1
        elif label[0] == 1:
            unSafeCount = unSafeCount + 1
        #print("safeCount = ",safeCount ,"unsafeCount = ",unSafeCount, "Duration1 = ", end1-start1, "Duration2 = ", end2-start2)
        print("[imagePrediction] ","safeCount = ",safeCount ,"unsafeCount = ",unSafeCount, "Duration = ", end2 - start1)
        totalTime = totalTime + (end2 - start1)

        if continuousSafetyCheckStartFlag == True:
            print ('time: ',totalTime ,'[continuousSafetyCheck] result(safe, unSafe): ' + str(safeCount) + ', ' + str(unSafeCount))
            if unSafeCount >= 1:
                print("continuousSafetyCheckT1CB: unsafey!!!!")
                continuousSafetyCheckResultUnsafePub.publish(True)
                continuousSafetyCheckStartFlag = False
                startCheck = False
   
    return


def chekc_elevator(msg):
    global safeCount, unSafeCount, startCheck, count, chekc_elevatorFlag,chekc_elevator_time_start, totalTime

    #if startCheck == False:
        #chekc_elevatorFlag = True
    chekc_elevator_time_start = time.time()
    print("[chekc_elevator]")
    safeCount = 0
    unSafeCount = 0
    count = 0
    totalTime = 0
    startCheck = True   
    t = Timer(3 ,chekc_elevatorCB)
    t.daemon = True
    t.start()
    
    return

def chekc_elevatorCB():
    global safeCount, unSafeCount, startCheck, checkCBPub, chekc_elevatorFlag, count,chekc_elevator_time_start, totalTime
    #print("chekc_elevatorCB safeCount =",safeCount)
    startCheck = False
    print ('[chekc_elevatorCB] result(safe, unSafe): ' + str(safeCount) + ', ' + str(unSafeCount))
    print ('[chekc_elevatorCB] count: ' + str(count) )
    
    chekc_elevator_time_end = time.time()
    
    if totalTime == 0:
        totalTime = chekc_elevator_time_end - chekc_elevator_time_start

    print ('[chekc_elevatorCB] time: ' + str(totalTime) )
    if count == 0:
        rospy.logerr("Count of Image is ZERO, Please check camera!!")
        rospy.loginfo("T0 report unsafe")
        #startCheck = False
        checkCBPub.publish(False)
        return

    checkResult = float(safeCount) / float(count - 1)
    if checkResult >= 0.9:
        checkCBPub.publish(True)
    else:
        checkCBPub.publish(False)
    
    # reset the startCheck flag
    
    #chekc_elevatorFlag = False

    return

def continuousSafetyCheckStart(msg):
    global safeCount, unSafeCount, startCheck, count ,startCheck,continuousSafetyCheckStartFlag, totalTime

    #if startCheck == False:
    continuousSafetyCheckStartFlag = True
    startCheck = True
    #record system time
    safeCount = 0
    unSafeCount = 0
    count = 0
    totalTime = 0

    t1 = Timer(0.5 ,continuousSafetyCheckT1CB)
    t1.daemon = True
    t1.start()

    t2 = Timer(3 ,continuousSafetyCheckT2CB)
    t2.daemon = True
    t2.start()

    return


def continuousSafetyCheckT1CB():
    global count

    if count == 0:
        rospy.logerr("Count of image is ZERO, Please check camera!!")
        rospy.loginfo("continuous T1 report unsafe")
        continuousSafetyCheckResultUnsafePub.publish(True)
    
    return

def continuousSafetyCheckT2CB():
    global startCheck , continuousSafetyCheckStartFlag
    #print ('[continuousSafetyCheckT1CB] result(safe, unSafe): ' + str(safeCount) + ', ' + str(unSafeCount))
    
    #if unSafeCount >= 1:
    #    print("continuousSafetyCheckT1CB: unsafey!!!!")
    #    continuousSafetyCheckResultUnsafePub.publish(True)
    
    startCheck = False
    continuousSafetyCheckStartFlag = False
    
    return
         

def main(args):
    rospy.init_node('ev_safty_check_test', anonymous=True)
    #image_sub = rospy.Subscriber("/camera_rear/tag_detections", Image, callback)
    image_sub = rospy.Subscriber("/usb_cam/image_rect_color", Image, imagePrediction, queue_size=1 ,buff_size=5000000)
    rospy.Subscriber('/checkEV',Bool,chekc_elevator)
    rospy.Subscriber('/continuousSafetyCheckStart', Bool, continuousSafetyCheckStart)

    #now = rospy.Time.now()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
