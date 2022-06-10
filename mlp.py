""" CS 545 Spring 2022
Final Project : Group Programming Assignment

Team Members:
Chris Anderson
xxxxxx
xxxxxx
xxxxxx

Analysis of X-Ray images classified as Normal, Covid, or Pneumonia
Experiments can vary the number of hidden layers, number of nodes per hidden layer,
activation function, and train/test split.
"""

import numpy as np
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
import os

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
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            masked_images.append(masked_image.flatten())

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

    # Get user input for the experiment parameters
    hidden = int(input("Enter the number of hidden layers: "))

    hiddenList = []
    for i in range(hidden):
        nextVal = int(input("Enter the number of nodes in the next hidden layer: "))
        hiddenList.append(nextVal)

    hiddenList = tuple(hiddenList)
    
    act = input("Enter the activation function you want to use. Choices are: identity, relu, logistic, or tanh. ").lower()
       
    testPct = float(input("Enter the percent of data in the TEST set. For example, enter 0.20 for 20% test, 80% train. "))    

    while (testPct > 1 or testPct < 0):
        testPct = input("Please enter a decimal number between 0 and 1.")
        try:
            testPct = float(testPct)
        except:
            print("Please enter a decimal number between 0 and 1.")
            
    # ===== Experiment 1 =====

    nn = NeuralNetwork()
    nn.load_data(COVID_IMAGE_PATH, COVID_MASK_PATH, 'covid')
    nn.load_data(PNEUMONIA_IMAGE_PATH, PNEUMONIA_MASK_PATH, 'pneumonia')
    nn.load_data(NORMAL_IMAGE_PATH, NORMAL_MASK_PATH, 'normal')


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
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(dataset, labels, test_size=testPct)

    # Scale the data using sklearn's built-in function
    sc_X = StandardScaler()
    X_trainscaled=sc_X.fit_transform(X_train)
    X_testscaled=sc_X.transform(X_test)



    # ===== Implement MLP Classifier =====
    clf = MLPClassifier(hidden_layer_sizes=(hiddenList),activation=act,random_state=1).fit(X_trainscaled, y_train)
    y_pred=clf.predict(X_testscaled)
    print(clf.score(X_testscaled, y_test))



    # ===== Analyze and Display Results =====
    
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels = ['Normal','Pneumonia','Covid'])
    cmd = cmd.plot(cmap=plt.cm.GnBu)
    title = f"Analysis of Lung X-Ray Images.\nActivation Function: {act}.\nTest Data Size: {testPct}. \
Test Data Size: {1-testPct}.\n{len(hiddenList)} Hidden Layers with Nodes: {hiddenList}"
    plt.title(title)
    plt.show()

