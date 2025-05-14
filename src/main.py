from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import numpy as np


if __name__ == '__main__':
    # resnet50.preprocess_input = resnet50.preprocess_input

    input_tensor = tf.keras.Input(shape=(224, 224, 3))

    model = ResNet50(weights='imagenet',
                     include_top=False,
                     pooling='avg',
                     input_shape=(224, 224, 3),
                     classes=200,
                     input_tensor = input_tensor,
                     classifier_activation='softmax')

    model.summary()
    print("Starting Main... ")