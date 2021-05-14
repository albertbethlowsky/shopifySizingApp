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
from time import time
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
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def getresults(dfs, X_test, y_test, X_train, y_train):
    #params_grid = [{'C': [1, 10, 100, 1000],'multi_class': ['crammer_singer']}]
    #high gamma means more curvature
    #low gamma means less curvature
    #degree, how many dimensions (features?)
    #C: degree of error. High C, many errors are allowed meaning many data points are allowed to cross the support vector
    #
    # Performing CV to tune parameters for best SVM fit 
    #svm_model = GridSearchCV(LinearSVC(), params_grid)
    svm_model = SVC(kernel='linear', decision_function_shape=dfs)
    start = time()
    svm_model.fit(X_train, y_train)
    time_libsvm = time()-start #seconds
    #clf = LinearSVC()
    #svm_model = SVC(kernel='rbf', gamma=1, degree=8)
    #degree is amount of dimensions aka features. only works with poly kernel 

    #random_state needs to be the same for each run or else different results will be computed
    #https://scikit-learn.org/stable/modules/cross_validation.html
    #why SVC is slow:
    #https://stackoverflow.com/questions/40077432/why-is-scikit-learn-svm-svc-extremely-slow
    #normalize the data, can improve the speed:
    #https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
    scores = cross_val_score(svm_model, X_test, y_test, cv=5)
    #svm_model.fit(X_train, y_train)

    #final_model = svm_model.best_estimator_
    y_pred = svm_model.predict(X_test)
    
    # View the accuracy score
    #print('Best score for training data:', svm_model.best_score_,"\n") 
    # View the best parameters for the model found using grid search
    #print('Best C:',svm_model.best_estimator_.C,"\n") 


    cm = pd.DataFrame(confusion_matrix(y_test,y_pred)).transpose()
    cmn = pd.DataFrame(confusion_matrix(y_test,y_pred, normalize='true')).transpose()
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()

    
    
    #print("Training set score for SVM: %f" % final_model.score(X_train, y_train))
    #print("Testing  set score for SVM: %f" % final_model.score(X_test, y_test ))
   
    #old:
    accuracy = accuracy_score(y_test, y_pred)
    kscore = scores.mean()
    kscore_stnd_dev = scores.std()
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # creating a confusion matrix
    #cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    
    name = 'Support Vector Machine (SVC)'
    dfsname = 'Decision Function Shape: ' + dfs
    return [[name,dfsname, accuracy, rmse, mae, kscore, kscore_stnd_dev, time_libsvm],df, cm, cmn]



