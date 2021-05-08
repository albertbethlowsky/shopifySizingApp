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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def getresults(X_train_chpt3, X_test_chpt3, y_train_chpt3, y_test_chpt3, X_train_runway, X_test_runway, y_train_runway, y_test_runway):
    gnb = GaussianNB().fit(X_train_runway, y_train_runway) #train it with runway
    

    gnb_predictions = gnb.predict(X_test_chpt3) #predict with chpt3
    
    # accuracy on X_test
   
    #print(accuracy)
    
    # creating a confusion matrix
    #cm = confusion_matrix(y_test_chpt3, gnb_predictions)
    scores = cross_val_score(gnb, X_train_runway, y_train_runway, cv=5)
    accuracy = accuracy_score(y_test_chpt3, gnb_predictions) #get prediction compared to actual sizes
    rmse = mean_squared_error(y_test_chpt3, gnb_predictions)
    mae = mean_absolute_error(y_test_chpt3, gnb_predictions)
    kscore = scores.mean()
    kscore_stnd_dev = scores.std()
    # print('Classification Report for KNN:')
    # print(classification_report(y_test_chpt3, y_pred, zero_division=1))

    name = 'Naive Bayes Classifier' 

    return [name, 'N/A', accuracy, rmse, mae, kscore, kscore_stnd_dev]

