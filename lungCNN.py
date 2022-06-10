""" CS 545 Spring 2022
Final Project : Group Programming Assignment
Team Members:
Chris Anderson
Steven Borrego
Emily Devlin
Max Schreyer
Analysis of X-Ray images classified as Normal, Covid, or Pneumonia
Experiments can vary the number of hidden layers, number of nodes per hidden layer,
activation function, and train/test split.
"""

import numpy as np
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

COVID_IMAGE_PATH = './COVID-19_Radiography_Dataset/COVID/images'
COVID_MASK_PATH = './COVID-19_Radiography_Dataset/COVID/masks'
PNEUMONIA_IMAGE_PATH = './COVID-19_Radiography_Dataset/Viral Pneumonia/images'
PNEUMONIA_MASK_PATH = './COVID-19_Radiography_Dataset/Viral Pneumonia/masks'
NORMAL_IMAGE_PATH = './COVID-19_Radiography_Dataset/Normal/images'
NORMAL_MASK_PATH = './COVID-19_Radiography_Dataset/Normal/masks'


class NeuralNetwork:
    def __init__(self):
        self.covid_masked_images = []
        self.pneumonia_masked_images = []
        self.normal_masked_images = []

    def load_data(self, image_path, mask_path, dataset):
        images = []
        masks = []
        masked_images = []

        # load images
        for filename in os.listdir(image_path):
            image = cv2.imread(os.path.join(image_path, filename))
            if image is not None:
                images.append(np.asarray(image))

        # load masks
        for filename in os.listdir(mask_path):
            mask = cv2.imread(os.path.join(mask_path, filename))
            if mask is not None:
                mask = cv2.resize(mask, (299, 299))
                masks.append(np.asarray(mask))

        # apply masks
        for i in range(0, len(images)):
            image = images[i]
            mask = masks[i]
            masked_image = cv2.bitwise_and(image, mask)
            masked_image = cv2.resize(masked_image, (70, 70)) / 255.0
            #masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            masked_images.append(masked_image)

        if dataset == 'covid':
            self.covid_masked_images = np.asarray(masked_images)
        elif dataset == 'pneumonia':
            self.pneumonia_masked_images = np.asarray(masked_images)
        elif dataset == 'normal':
            self.normal_masked_images = np.asarray(masked_images)


if __name__ == "__main__":
    print('====================================================')
    print('CS 545 Machine Learning - Group  Project')
    print('====================================================')

    testPct = 0.2

    # ===== Experiment 1 =====

    print("Loading Data...")
    nn = NeuralNetwork()
    nn.load_data(COVID_IMAGE_PATH, COVID_MASK_PATH, 'covid')
    nn.load_data(PNEUMONIA_IMAGE_PATH, PNEUMONIA_MASK_PATH, 'pneumonia')
    nn.load_data(NORMAL_IMAGE_PATH, NORMAL_MASK_PATH, 'normal')
    print("Data Processed")

    # ===== Separate Train and Test Sets =====
    # label encoding: normal = 0, covid = 2, pneumonia = 1

    # Create n x 1 label arrays for each set of images, n = # of data points for that label
    labels_covid = np.full((len(nn.covid_masked_images)), 2)
    labels_pneu = np.full((len(nn.pneumonia_masked_images)), 1)
    labels_normal = np.full((len(nn.normal_masked_images)), 0)

    # Concatenate all three label arrays into one array
    labels = np.hstack((labels_covid, labels_pneu, labels_normal))

    # Concatenate the data arrays into one array in the same order as labels
    dataset = np.vstack((nn.covid_masked_images, nn.pneumonia_masked_images, nn.normal_masked_images))

    # Divide the data into train and test using sklearn's built-in function
    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(dataset, labels, test_size=testPct)
    print("Data Separated")

    # ===== Preparing CNN model =====
    print("Preparing CNN")
    # Build the base of the CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(70, 70, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.summary()

    # Add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3))
    model.summary()

    print("CNN constructed")

    # Compiling and Training the model
    print("Training Model")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=2,
                        validation_data=(X_test, Y_test))

    yp_test = model.predict(X_test)
    yp_test = np.argmax(yp_test, axis=1)
    cm_test = confusion_matrix(Y_test, yp_test)
    t3 = ConfusionMatrixDisplay(cm_test)
    t3.plot()
    plt.show()
    plt.clf()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    print(test_acc)

