#NVIDIA model tested on CIFAR10 dataset.

'''
The CIFAR10 dataset contains 60,000 color images in 10 classes, 
with 6,000 images in each class. The dataset is divided into 50,000 
training images and 10,000 testing images. 
The classes are mutually exclusive and there is no overlap between them.
'''
'''
#22/05 Questions
1) Why is each epoch running to 1563 only?
2) How is it calculating gradient descent over here?
3) Need to plot loss graphs
'''

import tensorflow as tf
#import cv2
from tensorflow.keras import datasets, layers, models 
from tensorflow.keras.layers import Lambda
#from utils import INPUT_SHAPE
import matplotlib.pyplot as plt

#Download CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

#verify data
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
'''               
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
'''

#Data Pre-Processing
#Resizing CIFAR (32,32,3) to required input for method (66,200,3) 
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

train_images = tf.image.resize(train_images, [IMAGE_HEIGHT, IMAGE_WIDTH], method='gaussian')
test_images = tf.image.resize(test_images, [IMAGE_HEIGHT, IMAGE_WIDTH], method='gaussian')

#Create convolutional base
'''
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
'''
model = models.Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))  #Image-Normalisation layer - Avoids Saturation and make gradients work better
model.add(layers.Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))    #Filter size of 24 of 5*5 convolution of ELU and subsample with 2*2 
model.add(layers.Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
model.add(layers.Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='elu'))
model.add(layers.Conv2D(64, (3, 3), activation='elu'))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='elu'))
model.add(layers.Dense(50, activation='elu'))
model.add(layers.Dense(10, activation='elu'))
#model.add(Dense(1))

model.summary()

#Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

#Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
#Only achieves 0.6558 test_acc (65.58% basically)