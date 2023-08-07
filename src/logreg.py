# IMPORTING PACKAGES
# path tools
import os
# data loader
from tensorflow.keras.datasets import cifar10
# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# classification model
from sklearn.linear_model import LogisticRegression
# for saving models
from joblib import dump, load

import numpy as np
import cv2

# Loading data using function from TensorFlow
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # preparing data by assigning labels to the classes
    labels = ["airplane", 
              "automobile", 
              "bird", 
              "cat", 
              "deer", 
              "dog", 
              "frog", 
              "horse", 
              "ship", 
              "truck"]
    
    return X_train, y_train, X_test, y_test

# Doing preprocessing steps (converting to greyscale and scaling)
def preproc(X_train, X_test):
    # converting to greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # scaling the data
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0

    # reshaping training data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    # reshaping test data
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

    return X_train_dataset, X_test_dataset

# Training the logistic regression classifier
def logreg(X_train_dataset, y_train):
    # defining the classifier
    clf = LogisticRegression(penalty="none",
                             tol=0.1,
                             verbose=True,
                             solver="saga",
                             multi_class="multinomial").fit(X_train_dataset, y_train)
    # saving model to folder "models"
    clf_name = os.path.join("models", "logreg.joblib")
    dump(clf, clf_name)
    return clf

# Making a classification report
def clf_report(X_test_dataset, y_test, clf):
    labels = ["airplane", 
              "automobile", 
              "bird", 
              "cat", 
              "deer", 
              "dog", 
              "frog", 
              "horse", 
              "ship", 
              "truck"]
    y_pred = clf.predict(X_test_dataset)
    report = classification_report(y_test, y_pred, target_names=labels)

    # saving the report as a txt file
    report_path = os.path.join("reports", "logreg_report.txt")
    text_file = open(report_path, "w")
    text_file.write(report)
    text_file.close()


def main():
    X_train, y_train, X_test, y_test = load_data()
    X_train_dataset, X_test_dataset = preproc(X_train, X_test)
    clf = logreg(X_train_dataset, y_train)
    clf_report(X_test_dataset, y_test, clf) 

if __name__=="__main__":
    main()
