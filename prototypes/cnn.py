import os
import re
from keras_preprocessing.image.utils import validate_filename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, MaxPool2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import DirectoryIterator


def append_ext(fn):
    return str(str(fn) + ".png")


def main():
    df_train = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/trainLabels.csv"), dtype=str)
    df_test = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/sampleSubmission.csv"), dtype=str)

    df_train["id"] = df_train["id"].apply(append_ext)
    df_test["id"] = df_test["id"].apply(append_ext)

    # print(df_train.sample(5))

    train_dir = validation_dir = os.path.join(
        os.path.dirname(__file__), "../data/train/")
    test_dir = os.path.join(os.path.dirname(__file__), "../data/test/")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.4, 1.5],
        zoom_range=0.3,
        fill_mode="nearest"
    )
    # aug_iter = datagen.flow(df_train["id"][0], )

    # Spitting the dataset into training and validataion is completely optional
    train_generator = train_datagen.flow_from_dataframe(
        directory=train_dir,
        dataframe=df_train,
        x_col="id",
        y_col="label",
        subset="training",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(32, 32),
        validate_filenames=True)

    validation_generator = train_datagen.flow_from_dataframe(
        directory=validation_dir,
        dataframe=df_train,
        x_col="id",
        y_col="label",
        subset="validation",
        batch_size=32,
        shuffle=True,
        target_size=(32, 32),
        class_mode='categorical')

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_dataframe(
        directory=test_dir,
        dataframe=df_test,
        x_col="id",
        y_col=None,
        batch_size=32,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(32, 32)
    )

    # Model Building using Functiona API (Just dont use Sequential API[only linear models possible])
    # inputs = keras.Input(shape=(784,))
    img_input = keras.Input(shape=(32, 32, 3))

    # Block One
    x = Conv2D(filters=32, kernel_size=(3, 3),
               padding="same", activation="relu")(img_input)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Block Two
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(units=512, activation="relu")(x)
    x = Dropout(0.5)(x)
    class_output = Dense(units=512, activation="softmax")(x)

    model = Model(inputs=img_input, outputs=class_output)
    print(model.summary())
    keras.utils.plot_model(model, os.path.join(
        os.path.dirname(__file__), "../output/convnet1.png"), show_shapes=True)


if __name__ == '__main__':
    main()
