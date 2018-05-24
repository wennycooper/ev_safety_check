#!/usr/bin/env python
# license removed for brevity
import rospy
import rospkg
import cv2
import shutil, sys, termios, tty, os, time
import numpy as np

rospack = rospkg.RosPack()
packPath = rospack.get_path('ev_safety_check')
print packPath

trainUnlabeledDir = packPath + "/train/unlabeled/"
trainSafeDir = packPath + "/train/safe/"
trainUnsafeDir = packPath + "/train/unsafe/"
fullpath = os.path.join

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
 
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

button_delay = 0.2
def image_label_tool():
    rospy.init_node('image_label_tool', anonymous=True)

    cv2.namedWindow("Image")

    for fname in sorted(os.listdir(trainUnlabeledDir)):
        if fname.endswith(".jpeg"):
            print 'filename: ', trainUnlabeledDir+fname
            im = cv2.imread(trainUnlabeledDir+fname)
            cv2.imshow("Image", im)
            cv2.waitKey(10)
            
            char = "0"
            # comment out this to playback and not label
            char = getch()
 
            if (char == "q"):
                print("Quit!")
                exit(0)
 
            elif (char == "l"):
                print("label to unsafe")
                move_file(fullpath(trainUnlabeledDir, fname), trainUnsafeDir)
                time.sleep(button_delay)
 
            elif (char == "j"):
                print("label to safe!")
                move_file(fullpath(trainUnlabeledDir, fname), trainSafeDir)
                time.sleep(button_delay)
 
def move_file(filePath, targetPath):
    # print 'in move_file():'
    # print filePath
    # print targetPath
    print 'mv ', filePath, ' to ', targetPath
    shutil.move(filePath, targetPath)

if __name__ == '__main__':

    print 'J: label to safe,  L: label to unsafe, Q: quit, others: ignore'
    try:
        image_label_tool()
    except rospy.ROSInterruptException:
        pass
