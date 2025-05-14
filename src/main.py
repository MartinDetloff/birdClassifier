from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import numpy as np
# tf.enable_eager_execution()



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

    data_test_dir = "C:/Users/Martin/Documents/GitHub/birdClassifier/data/archive/Test"
    data_train_dir = "C:/Users/Martin/Documents/GitHub/birdClassifier/data/archive/Train"
    image_size = (224, 224)
    batch_size = 32




    test_dataset = tf.keras.utils.image_dataset_from_directory(
        data_test_dir, labels='inferred', image_size= image_size, batch_size=batch_size
    )

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_train_dir, labels='inferred', image_size= image_size, batch_size=batch_size
    )


    print_class_labels(test_dataset, True)
    print_class_labels(train_dataset, False)



    print(test_dataset)

    # resnet50.preprocess_input = resnet50.preprocess_input

    # input_tensor = tf.keras.Input(shape=(224, 224, 3))
    #
    # model = ResNet50(weights='imagenet',
    #                  include_top=False,
    #                  pooling='avg',
    #                  input_shape=(224, 224, 3),
    #                  classes=200,
    #                  input_tensor = input_tensor,
    #                  classifier_activation='softmax')



    # model.summary()
    print("Starting Main... ")



    tf.keras.preprocessing.image_dataset_from_directory(

    )