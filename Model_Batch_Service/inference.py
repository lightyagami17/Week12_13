#!/usr/bin/python3
# inference.py


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import joblib



def inference():

    MODEL_DIR = './drive/MyDrive/Week12_13/Model_Online_Service/'
    MODEL_PATH_LDA = 'Lda.sav'
    MODEL_PATH_NN = 'NN.sav'
    MODEL_PATH_SVM = 'svm.joblib'
        
    # Load, read and normalize training data
    testing = "test.csv"
    data_test = pd.read_csv("./drive/MyDrive/Week12_13/Model_Batch_Service/test.csv")
        
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)

    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')
    clf_lda = joblib.load(os.path.join(MODEL_DIR, MODEL_PATH_LDA))
    clf_nn = joblib.load(os.path.join(MODEL_DIR, MODEL_PATH_NN))
    clf_svm = joblib.load(os.path.join(MODEL_DIR, MODEL_PATH_SVM))
    
    # Models training
    
    # Run model
    print(MODEL_PATH_LDA)
    clf_lda = load(MODEL_PATH_LDA)
    print("LDA score and classification:")
    prediction_lda = clf_lda.predict(X_test)
    report_lda = classification_report(y_test, prediction_lda)

    print(clf_lda.score(X_test, y_test))
    print('LDA Prediction:', prediction_lda)
    print('LDA Classification Report:', report_lda)

        
    # Run model
    clf_nn = load(MODEL_PATH_NN)
    print("NN score and classification:")
    prediction_nn = clf_nn.predict(X_test)
    report_nn = classification_report(y_test, prediction_nn)


    print(clf_nn.score(X_test, y_test))
    print('NN Prediction:', prediction_nn)
    print('NN Classification Report:', report_nn)

    clf_svm = load(os.path.join(MODEL_DIR, MODEL_PATH_SVM))
    print("SVM score and classification:")
    prediction_svm = clf_svm.predict(X_test)
    report_svm = classification_report(y_test, prediction_svm)

    print(clf_svm.score(X_test, y_test))
    print('SVM Prediction:', prediction_svm)
    print('SVM Classification Report:', report_svm)
    
    
if __name__ == '__main__':
    inference()




