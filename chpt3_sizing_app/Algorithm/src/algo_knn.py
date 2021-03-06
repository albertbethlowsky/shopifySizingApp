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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def getresults(weight, k, X_test, y_test, X_train, y_train):
    #Then define the model using KNeighborsClassifier and fit the train data in the model:
    #Define the model: Init K-NN

    #k = int(math.sqrt(len(X_train_runway)))#sqrt of n, seems to be good k value. k must be uneven, such that the result wont be even between two clusters 

    if (k % 2) == 0:  
        k = k-1 #if k is even, subtract one. 
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean', weights=weight) #euclidean, finds the 
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    kscore = scores.mean()
    kscore_stnd_dev = scores.std()
    #N_neighbors here is 'K' 
    #p is the power parameter to define the metric used, which 'Euclidean' in our case
    start = time()
    classifier.fit(X_train, y_train)
    time_knn = time()-start #seconds

    #predict the test set results
    y_pred = classifier.predict(X_test)
    print("this is X_test:")
    print(X_test)
    print("this is KNN y_pred:")
    print(y_pred)

    #evaluate model
    #cm = confusion_matrix(y_test, y_pred)

    s = 'k= '+str(k) + ' weight=' + weight

    
    accuracy = accuracy_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # print('Classification Report for KNN:')
    # print(classification_report(y_test, y_pred, zero_division=1))
    cm = pd.DataFrame(confusion_matrix(y_test,y_pred)).transpose()
    cmn = pd.DataFrame(confusion_matrix(y_test,y_pred, normalize='true')).transpose()
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()

    name = 'K-Nearest Neighbor'

    return [[name, s, accuracy, rmse, mae, kscore, kscore_stnd_dev, time_knn],df, cm, cmn]