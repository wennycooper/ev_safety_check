#!/usr/bin/env python
import rospy
import rospkg
import sys
import os
import numpy as np
import tensorflow as tf
from input_data import input_data
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import load_model

np.random.seed(10)

rospack = rospkg.RosPack()
packPath = rospack.get_path('ev_safety_check')
print packPath

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image)
    plt.show()

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()


def main(argv):
    myData = input_data()
    train_images, train_labels = myData.read_data_sets()

    #print train_images.shape
    #print train_labels.shape
    #plot_image(train_images[0])

    train_images_4d=train_images.reshape(train_images.shape[0],48,64,3).astype('float32')
    train_images_4d_normalized = train_images_4d / 255
    train_labels_onehot = np_utils.to_categorical(train_labels)

    model = Sequential()

    model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(48,64,3), 
                 activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(2,activation='softmax'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy']) 

    train_history=model.fit(x=train_images_4d_normalized, 
                            y=train_labels_onehot,
                            validation_split=0.2, 
                            epochs=20, batch_size=128,verbose=2)

    # comment this if you are not in console
    # show_train_history(train_history, 'acc', 'val_acc')


    # save model
    modelFileName = packPath + "/models/my_model.h5"
    model.save(modelFileName)

    pass

if __name__ == '__main__':
    main(sys.argv)


