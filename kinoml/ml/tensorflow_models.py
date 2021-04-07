"""
Example models for TensorFlow

.. note::

    This code is not currently in use.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten


def DNN(input_dim):
    """
    DNN builds and compiles a TF model (a Deep Neural Network) that takes as input 'input_dim'
    Parameters
    ==========
    input_dim : tuple of int
        Expected shape of the input data
    Returns
    =======
    model : tf.keras.models.Sequential
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(350, activation="relu", input_dim=input_dim),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def CNN(input_shape):
    """
    CNN builds and compiles a TF model (a Convolutional Neural Network) that takes as input 'input_shape'
    Parameters
    ==========
    input_shape : tuple of int
        Expected shape of the input data
    Returns
    =======
    model : tf.keras.models.Sequential
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=3,
                activation="relu",
                padding="same",
                input_shape=input_shape,
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def MPNN(input_shape):
    """
    MPNN builds and compiles a TF model (a Message Passing Neural Network) that takes as input 'input_shape'
    Parameters
    ==========
    input_shape : tuple of int
        Expected shape of the input data
    Returns
    =======
    model : tf.keras.models.Sequential
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=3,
                activation="relu",
                padding="same",
                input_shape=input_shape,
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
