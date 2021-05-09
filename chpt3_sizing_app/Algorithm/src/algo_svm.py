import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import json
from sklearn.model_selection import train_test_split # splits the data, part is training and part is testing
from sklearn.preprocessing import StandardScaler # So we dont have bias of high numbers. Uniformed of -1 and 1
from sklearn.neighbors import KNeighborsClassifier #the actual tool
#three tools to test:
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score
from termcolor import colored as cl # elegant printing of text
import seaborn as sb # visualizations
import matplotlib.pyplot as plt # editing visualizations
import math
from matplotlib import style # setting styles for plots
from sklearn.metrics import accuracy_score # algorithm accuracy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def getresults(X_test, y_test, X_train, y_train):
    clf = LinearSVC()
    #random_state needs to be the same for each run or else different results will be computed
    #https://scikit-learn.org/stable/modules/cross_validation.html
    #why SVC is slow:
    #https://stackoverflow.com/questions/40077432/why-is-scikit-learn-svm-svc-extremely-slow
    #normalize the data, can improve the speed:
    #https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    clf.fit(X_train, y_train)
    svm_predictions = clf.predict(X_test)
    # model accuracy for X_test  
    accuracy = accuracy_score(y_test, svm_predictions)
    kscore = scores.mean()
    kscore_stnd_dev = scores.std()
    rmse = mean_squared_error(y_test, svm_predictions)
    mae = mean_absolute_error(y_test, svm_predictions)
    
    # creating a confusion matrix
    #cm = confusion_matrix(y_test, svm_predictions)
    #print(cm)
    name = 'Support Vector Machine'

    return [name,'N/A', accuracy, rmse, mae, kscore, kscore_stnd_dev]



