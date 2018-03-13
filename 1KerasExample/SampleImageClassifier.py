import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

import numpy as np
import cv2
import os

IMAGE_FOLDER_NAME = "imageData"
TRAIN_FOLDER_NAME = "train"
TEST_FOLDER_NAME = "test"
DEV_FOLDER_NAME = "dev"

# number of classes to train
num_classes = 10

batch_size = 2

epochs = 20

test = np.arange(24).reshape((2,3,4))
test = np.compress([0, 1], test, axis=2)
print(test.shape[0])
print(test.shape[1])

def readImages(dbpath, image_type=".png", colored=True) :
    imgs = []
    labels = []
    for filename in os.listdir(dbpath):
        if filename.endswith(image_type):
            filepath = os.path.join(dbpath, filename)
            print("Reading file - ", filepath)
            img = cv2.imread(filepath)
            if colored :
                # compress rgb
                img = np.compress([0, 1], img, axis=2)
            img = img.reshape(img.shape[0] * img.shape[1])
            # compress values in [0;1] interval
            img = img / 255
            imgs.append(img)
            label = filename.split("_")[1]
            labels.append(label)
    return np.array(imgs), np.array(labels)

imagePath = os.path.join(os.path.curdir, IMAGE_FOLDER_NAME)

train_data, train_lables = readImages(os.path.join(imagePath, TRAIN_FOLDER_NAME))
dev_data, dev_labels = readImages(os.path.join(imagePath, DEV_FOLDER_NAME))
test_data, test_labels = readImages(os.path.join(imagePath, TEST_FOLDER_NAME))

# create one-hot vector from labels
train_lables = keras.utils.to_categorical(train_lables, num_classes)
dev_labels = keras.utils.to_categorical(dev_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(16384,)))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(train_data, train_lables,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(dev_data, dev_labels))
score = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

