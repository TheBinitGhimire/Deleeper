#!/usr/bin/env python3

import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import random, shutil

from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.models import load_model

from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization

import matplotlib.pyplot as plt 
import numpy as np

# This file is used to train the model.

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

# Defining the Batch Size and Target Size!
batchSize = 32
targetSize = (24, 24)

path = os.getcwd()
if os.path.exists(os.path.join(path, "data", ".gitkeep")):
    print("The model is already trained. So, the data hasn't been made available.")
    os._exit(1)
    
trainBatch = generator('data/train', shuffle=True, batch_size=batchSize, target_size=targetSize)
validBatch = generator('data/valid', shuffle=True, batch_size=batchSize, target_size=targetSize)

SPE = len(trainBatch.classes)//batchSize
VS = len(validBatch.classes)//batchSize
print(SPE, VS)

# Model Information!
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Model Compilation!
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(trainBatch, validation_data=validBatch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

# Saving the Model!
model.save('models/model.h5', overwrite=True)