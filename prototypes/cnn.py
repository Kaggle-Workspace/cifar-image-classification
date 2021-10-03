import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, MaxPool2D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def append_ext(fn):
    return str(str(fn) + ".png")


def main():
    fix_gpu()
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
    class_output = Dense(units=10, activation="softmax")(x)

    model = Model(inputs=img_input, outputs=class_output)
    print(model.summary())
    model.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    keras.utils.plot_model(model, os.path.join(
        os.path.dirname(__file__), "../output/convnet1.png"), show_shapes=True)

    # Setting the step size
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
    STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

    # model.fit(
    #     train_generator,
    #     steps_per_epoch=2000,
    #     epochs=50,
    #     validation_data=validation_generator,
    #     validation_steps=800
    # )

    model.fit(train_generator,
              steps_per_epoch=STEP_SIZE_TRAIN,
              validation_data=validation_generator,
              validation_steps=STEP_SIZE_VALID,
              epochs=50
              )


if __name__ == '__main__':
    main()
