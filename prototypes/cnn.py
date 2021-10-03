import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, MaxPool2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def append_ext(fn):
    return str(str(fn) + ".png")


def load_train_images():
    df_train = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/trainLabels.csv"), dtype=str)
    df_valid = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/sampleSubmission.csv"), dtype=str)

    df_train["id"] = df_train["id"].apply(append_ext)
    df_valid["id"] = df_valid["id"].apply(append_ext)

    # print(df_train.sample(5))

    train_dir = os.path.join(os.path.dirname(__file__), "../data/train/")
    validation_dir = os.path.join(os.path.dirname(__file__), "../data/test/")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.25,
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
        target_size=(32, 32),
        class_mode='categorical')

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)

    # test_generator = test_datagen.flow_from_dataframe()


def main():
    load_train_images()


if __name__ == '__main__':
    main()
