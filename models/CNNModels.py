import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, MaxPool2D, AveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNN1(keras.models.Model):

    def __init__(self):
        super(CNN1,self).__init__()
        # Model Building using Functiona API (Just dont use Sequential API[only linear models possible])
        # inputs = keras.Input(shape=(784,))
        img_input = keras.Input(shape=(32, 32, 3))

        # Block One
        self.conv2d_1 =  Conv2D(filters=32, kernel_size=(3, 3),
                   padding="same", activation="relu")
        self.conv2d_2 =  Conv2D(32, (3, 3), activation="relu")
        self.avg_pool_2d_1 =  AveragePooling2D(pool_size=(2, 2))
        self.dropout_1 =  Dropout(0.25)

        # Block Two
        # self.conv2d_3 =  Conv2D(64, (3, 3), padding="same", activation="relu")
        self.conv2d_4 =  Conv2D(64, (3, 3), activation="relu")
        self.avg_pool_2d_2 =  AveragePooling2D(pool_size=(2, 2))
        self.dropout_2 =  Dropout(0.25)

        self.flatten_1 =  Flatten()
        self.dense_1 =  Dense(units=512, activation="relu")
        self.dropout_1 =  Dropout(0.5)
        class_output = Dense(units=10, activation="softmax")

    def call(self,inputs):
        pass
