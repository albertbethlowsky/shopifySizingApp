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
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


def chpt3vals(gender, testsize):
    #chpt3:
    dataset = pd.read_csv('./Data/clean_chpt3.csv')
    dataset.age = dataset.age.astype(np.float64)
    dataset.bmi = dataset.bmi.astype(np.float64)
    dataset = dataset[~dataset['gender'].isin(gender)] #isolate male and female
        
    #label the data, such that strings get an int representation
    le.fit(dataset.fit)
    fit_label = le.transform(dataset.fit)
    #large=1
    #fit=0
    #small=2

    

    le.fit(dataset.body_type)
    body_type_label = le.transform(dataset.body_type)

    le.fit(dataset.product_category)
    product_category_label = le.transform(dataset.product_category)

    le.fit(dataset.bust_size_cat)
    bust_size_cat_label = le.transform(dataset.bust_size_cat)

    dataset.fit                     = fit_label
    dataset.body_type               = body_type_label
    dataset.product_category        = product_category_label
    dataset.bust_size_cat           = bust_size_cat_label
    
    #get list of unique values
    dataset = dataset[['fit', 'product_size', 'bust_size_num_eu', 'bust_size_cat', 'height_meters', 'weight_kg', 'product_category', 'age', 'body_type']].copy()
    dataset.product_size = dataset.product_size.astype(np.int64)
    dataset.age = dataset.age.astype(np.int64)
    
    dataset = dataset.sample(frac=1) #shuffle the dataset'
    
    #plot_correlation(dataset)
    X = dataset.iloc[:,1:8] 
    y = dataset.iloc[:,0] #fit

    # print('this is chpt3 x')
    # print(X.head(20))
    # print(X.describe())
    # X.info()
    # print('this is chpt3 y:')
    # print(y.head(20))
    # print(y.describe())
   
    # print(X)
    # print(y)
    if(testsize<0.50):
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size=testsize)
        #Rule of thumb: any algorithm that computes distance or assumes normality, scale your features!
        #feature sacling
        sc_X = StandardScaler()

        X_train=sc_X.fit_transform(X_train)

        X_test=sc_X.transform(X_test)
        return [X_train, X_test, y_train, y_test]
    else:
        #Rule of thumb: any algorithm that computes distance or assumes normality, scale your features!
        #feature sacling
        sc_X = StandardScaler()
       
        X_train=sc_X.fit_transform(X)
       
        #using all of chpt3 
        X_test=sc_X.transform(X)

        y_test=y.sample(frac=1) #shuffle y
        y_train=y.sample(frac=1)
        return [X_train, X_test, y_train, y_test]

