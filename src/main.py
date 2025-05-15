from keras import Sequential
from keras.src.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import numpy as np
import kagglehub


# Method to print out the class labels
def print_class_labels(dataset, isTest):
    if isTest:
        print("Test Data Class Labels")

    else:
        print("Train Data Class Labels")

    for i, class_label in enumerate(dataset.class_names):
        print("This is the class label:", i , " ",  class_label)

    print("-------------------------------------------------------")


if __name__ == '__main__':
    print("Starting Main... ")

    print("TensorFlow version:", tf.__version__)
    print("GPUs available:", tf.config.list_physical_devices('GPU'))

    # Use kaggle's API to download the dataset from the website
    path = kagglehub.dataset_download("kedarsai/bird-species-classification-220-categories")

    # print("Path to dataset files:", path)

    # specify the image size and the batch size to be used
    image_size = (224, 224)
    batch_size = 32

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        path + "/test", labels='inferred', image_size= image_size, batch_size=batch_size
    )

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        path + "/train", labels='inferred', image_size= image_size, batch_size=batch_size
    )


    print_class_labels(test_dataset, True)
    print_class_labels(train_dataset, False)

    resnet_model = Sequential()

    # we want to start our restNet50 model here with our own input/ output layers
    model = ResNet50(weights='imagenet',
                     include_top=False,
                     pooling=None,
                     input_shape=(224, 224, 3),
                     input_tensor=None,
                     classes = 200
                     )

    # for layer in model.layers:
    #     layer.trainable = False

    resnet_model.add(model)
    resnet_model.add(GlobalAveragePooling2D())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(200, activation='softmax'))

    resnet_model.summary()

    resnet_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    epochs = 10
    history = resnet_model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    resnet_model.save('my_model.keras')

    print(test_dataset)



