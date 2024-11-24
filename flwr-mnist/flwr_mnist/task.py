"""flwr-mnist: A Flower / TensorFlow app."""

import os
import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    # Initialize the input
    input_layer = tf.keras.Input(shape=(28, 28, 1))

    # First convolutional layer
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    # Second convolutional layer
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Dropout to prevent overfitting
    dropout1 = layers.Dropout(0.25)(pool2)

    # Flatten the 3D tensor to 1D
    flatten = layers.Flatten()(dropout1)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dropout2 = layers.Dropout(0.5)(dense1)

    # Output layer for 10 classes
    output_layer = layers.Dense(10, activation='softmax')(dropout2)

    # Define the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


fds = None  # Cache FederatedDataset


def load_data():
    (x_train , y_train),(x_test , y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255
    return x_train, y_train, x_test, y_test
