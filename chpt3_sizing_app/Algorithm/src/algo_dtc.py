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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def getresults(maxdepth, X_test, y_test, X_train, y_train):
    #insert runway's var:
    start = time()
    dtree_model = DecisionTreeClassifier(max_depth = maxdepth).fit(X_train, y_train)
    time_dtc = time()-start #seconds

    #insert chpt3's var:
    y_pred = dtree_model.predict(X_test)

    #print(cm)
    scores = cross_val_score(dtree_model, X_test, y_test, cv=5)
    accuracy = accuracy_score(y_test, y_pred)
    rmse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    kscore = scores.mean()
    kscore_stnd_dev = scores.std()
    name = 'Decision Tree Classifier'
    s = 'max_depth= '+str(maxdepth)
    cm = pd.DataFrame(confusion_matrix(y_test,y_pred)).transpose()
    cmn = pd.DataFrame(confusion_matrix(y_test,y_pred, normalize='true')).transpose()
    
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    return [[name, s, accuracy, rmse, mae, kscore, kscore_stnd_dev, time_dtc],df, cm, cmn]
    # print(s)


