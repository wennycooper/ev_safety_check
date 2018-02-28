import os
import sys
import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)

import matplotlib.pyplot as plt


trainSafeDir = "/home/kkuei/catkin_ws/src/ev_safety_check/train/safe/"
trainUnsafeDir = "/home/kkuei/catkin_ws/src/ev_safety_check/train/unsafe/"
testSafeDir = "/home/kkuei/catkin_ws/src/ev_safety_check/test/safe/"
testUnsafeDir = "/home/kkuei/catkin_ws/src/ev_safety_check/test/unsafe/"


class input_data:
    def __init__(self):
        self.num_train_examples_ = 0
        self.train_images = np.zeros((0,48,64,3))
        self.train_labels = np.zeros((0))
        
        self.num_test_examples_ = 0
        self.count = 0
        pass

    def plot_image(self,image):
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.imshow(image)
        plt.show()


    def read_data_sets(self):
        # read training dataset to np.ndarry
        
        for fname in os.listdir(trainSafeDir):
            if fname.endswith(".jpeg"):
                self.count = self.count + 1
                im = cv2.imread(trainSafeDir+fname)
                #print self.count, im.shape, type(im)
                im1 = im.reshape((1,48,64,3))
                self.train_images = np.concatenate((self.train_images, im1), axis=0) #put image in ndarray
                self.train_labels = np.concatenate((self.train_labels, [1]), axis=0) #label=1
                #print self.count, self.train_images.shape
            else:
                pass
        for fname in os.listdir(trainUnsafeDir):
            if fname.endswith(".jpeg"):
                self.count = self.count + 1
                im = cv2.imread(trainUnsafeDir+fname)
                #print self.count, im.shape, type(im)
                im1 = im.reshape((1,48,64,3))
                self.train_images = np.concatenate((self.train_images, im1), axis=0)
                self.train_labels = np.concatenate((self.train_labels, [0]), axis=0) #label=0
                #print self.count, self.train_images.shape
            else:
                pass

        #print "read images completed"
        #print "train_images.shape = ", self.train_images.shape
        #print "train_labels.shape = ", self.train_labels.shape

        #rgb = cv2.cvtColor(self.train_images[0], cv2.COLOR_BGR2RGB)
        #self.plot_image(rgb)
        #self.plot_image(self.train_images[0])

        #return self.train_images, self.train_labels

        # shuffle arrays
        s = np.arange(self.train_images.shape[0])
        np.random.shuffle(s)
        self.train_images_ = self.train_images[s]
        self.train_labels_ = self.train_labels[s]
        return self.train_images_, self.train_labels_


