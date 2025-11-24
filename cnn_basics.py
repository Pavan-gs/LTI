# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:43:59 2021

@author: Deepstrats
"""

import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


# Convolutional Neural Network : CNN

# Convolutional layer : Convolve/ convolution

# Convolution : Linear Operation where multiplication of set of weights with the input data takes place

# Kernel/Filter/Feature Detector : Weights

# Feature Map : o/p of one filter applied to the previous layer

# Channels : Gray scale(2), RGB (3)

# Max pooling

# Flattening

# Fully connected neural network

# Zero padding

# Gray scale image : 32*32*1 = 1024 pixels
# Color image = 32*32*3 

from keras.datasets import mnist, fashion_mnist, cifar10
#m = mnist.load_data()
#c = cifar10.load_data()
#mnist.load_data()

#(x_train,y_train),(x_test,y_test) = mnist.load_data()
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
for i in range(9):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.show()

x_train.shape
x_train.shape[0]
x_test.shape[0]
x_train = x_train.reshape((x_train.shape[0],28,28,1))
x_test = x_test.reshape((x_test.shape[0],28,28,1))

x_train[0]

#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train

train_norm = x_train.astype('float32')
test_norm = x_test.astype('float32')

train_norm = train_norm/255.0
test_norm = test_norm/255.0

from keras.models import Sequential
from keras.layers import Dense , Dropout, BatchNormalization

from keras.layers import Conv1D,Conv2D, Conv3D, MaxPooling2D, Flatten

model = Sequential()

model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(100,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
m2 = model.fit(train_norm,y_train, epochs=10, batch_size=25,validation_data=(test_norm,y_test))

pred = model.predict(test_norm)

from sklearn.metrics import mean_squared_error, confusion_matrix

pred[0]
np.argmax(pred[0])
y_test[0]

model.summary()

acc=m2.history['accuracy']
val_acc=m2.history['val_accuracy']

loss=m2.history['loss']
val_loss=m2.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()

from keras.preprocessing.image import load_img, img_to_array
img = load_img('D:/Deloitte/Ds5Rc.png', grayscale=True, target_size=(28,28))
img
img = img_to_array(img)
img = img.reshape(1,28,28,1)
img = img.astype('float32')
img = img/255.0

pred_digit = model.predict(img)
pred_digit.argmax()
pred_digit.max()

img = load_img('D:/Deloitte/th.jpg', grayscale=True, target_size=(28,28))
img
img = img_to_array(img)
img = img.reshape(1,28,28,1)
img = img.astype('float32')
img = img/255.0

pred_digit = model.predict(img)
pred_digit.argmax()
pred_digit.max()

img = load_img('D:/Deloitte/MNIST_image_7.png', grayscale=True, target_size=(28,28))
img
img = img_to_array(img)
img = img.reshape(1,28,28,1)
img = img.astype('float32')
img = img/255.0

pred_digit = model.predict(img)
pred_digit.argmax()
pred_digit.max()






































