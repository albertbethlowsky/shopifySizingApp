import numpy as np
import pandas as pd
import seaborn as sns
from time import time
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

def getresults(X_test, y_test, X_train, y_train):
    start = time()
    gnb = GaussianNB().fit(X_train, y_train) #train it with runway
    time_nbc = time()-start #seconds

    y_pred = gnb.predict(X_test) #predict with chpt3
    
    # accuracy on X_test
   
    #print(accuracy)
    
    # creating a confusion matrix
    #cm = confusion_matrix(y_test_chpt3, gnb_predictions)
    scores = cross_val_score(gnb, X_test, y_test, cv=5)
    accuracy = accuracy_score(y_test, y_pred) #get prediction compared to actual sizes
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    kscore = scores.mean()
    kscore_stnd_dev = scores.std()
   
    cm = pd.DataFrame(confusion_matrix(y_test,y_pred)).transpose()
    cmn = pd.DataFrame(confusion_matrix(y_test,y_pred, normalize='true')).transpose()
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()

    name = 'Naive Bayes Classifier' 

    return [[name, 'N/A', accuracy, rmse, mae, kscore, kscore_stnd_dev, time_nbc],df, cm, cmn]

