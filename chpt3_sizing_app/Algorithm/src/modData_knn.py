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
import treat_RunWay as treatrw
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

# Read dataset to pandas dataframe
#dataset = pd.read_csv('realUsers.csv')

#Clean data:
#treatrw.create_csv('C:/Users/Frederik/Desktop/shopifySizingApp/chpt3_sizing_app/Algorithm/Data/renttherunway_final_data.json', 'C:/Users/Frederik/Desktop/shopifySizingApp/chpt3_sizing_app/Algorithm/Data/clean_runway.csv')

#Cleaned data:
#runway:
#dataset = pd.read_csv('C:/Users/Frederik/Desktop/shopifySizingApp/chpt3_sizing_app/Algorithm/Data/clean_runway.csv')

#chpt3:
dataset = pd.read_csv('C:/Users/Frederik/Desktop/shopifySizingApp/chpt3_sizing_app/Algorithm/Data/clean_chpt3.csv')

#dataset.info()
#https://www.kaggle.com/agrawaladitya/step-by-step-data-preprocessing-eda?scriptVersionId=5377677


#KNN - predict whether a person is xs,s,m,l or xl

#label the data, such that strings get an int representation
le.fit(dataset.fit)
fit_label = le.transform(dataset.fit)

le.fit(dataset.body_type)
body_type_label = le.transform(dataset.body_type)

le.fit(dataset.product_category)
product_category_label = le.transform(dataset.product_category)

le.fit(dataset.product_size)
product_size_label = le.transform(dataset.product_size)

le.fit(dataset.bust_size_cat)
bust_size_cat_label = le.transform(dataset.bust_size_cat)

dataset.fit                     = fit_label
dataset.body_type               = body_type_label
dataset.product_category        = product_category_label
dataset.product_size            = product_size_label
dataset.bust_size_cat           = bust_size_cat_label


#reverse the labels:
#ori_cup_size_label = le.inverse_transform(cup_size_label)

def plot_correlation(data):
    '''
    plot correlation's matrix to explore dependency between features 
    '''
    # init figure size
    #rcParams['figure.figsize']=15,20
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.show()
    fig.savefig('corr.png')

#get list of unique values
dataset = dataset[['fit','product_size','body_type', 'age', 'bust_size_num', 'bust_size_cat', 'bmi', 'product_category']].copy()


missing_data = pd.DataFrame({'total_missing': dataset.isnull().sum(), 'perc_missing': (dataset.isnull().sum()/158220)*100})
print(missing_data)
dataset.info()
print(dataset.head(20))

X = dataset.iloc[:,1:7] 
y = dataset.iloc[:,0] #fit

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size=0.2)

#Rule of thumb: any algorithm that computes distance or assumes normality, scale your features!
#feature sacling
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Then define the model using KNeighborsClassifier and fit the train data in the model:
#Define the model: Init K-NN
k = int(math.sqrt(len(X_train)))#sqrt of n, seems to be good k value. k must be uneven, such that the result wont be even between two clusters 

if (k % 2) == 0:  
    k = k-1 #if k is even, subtract one. 
classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean') #euclidean, finds the distance between test data point and train data point
#N_neighbors here is 'K' 
#p is the power parameter to define the metric used, which 'Euclidean' in our case
classifier.fit(X_train, y_train)

#predict the test set results
y_pred = classifier.predict(X_test)

#evaluate model
cm = confusion_matrix(y_test, y_pred)

s = 'k= '+str(k)
users = len(dataset)

print('Accuracy:')
print(accuracy_score(y_test, y_pred))
print('RMSE:')
print(mean_squared_error(y_test, y_pred))
print('MAE:')
print(mean_absolute_error(y_test, y_pred))
print('Classification Report for KNN:')
print(classification_report(y_test, y_pred, zero_division=1))

#import src.appendToCsv as atc
# row_contents = ['K-Nearest Neighbor',s,users,id,accuracy_score(y_test, y_pred),mean_squared_error(y_test,y_pred), mean_absolute_error(y_test,y_pred)]
# atc.append_list_as_row('results/MLresults_modData.csv', row_contents)

# for i in range(1,105,2):
#     knn = KNeighborsClassifier(n_neighbors=i, p=2, metric='euclidean')
#     #Train the model using the training sets
#     knn.fit(X_train, y_train)
#     #Predict the response for test dataset
#     y_pred = knn.predict(X_test)
#     print('Accuracy:',accuracy_score(y_test, y_pred), 'for ',i)
#     print (cm)

#     print('Root Mean Squared Error (RMSE):')
#     print(mean_squared_error(y_test,y_pred))
#     print('Mean Aboslute Error (MAE):')
#     print(mean_absolute_error(y_test,y_pred))
#     print('Accuracy Score:')
# # print(f1_score(y_test,y_pred))
#     print(accuracy_score(y_test, y_pred))

#     result1 = classification_report(y_test, y_pred)
#     print('Classification Report:')
#     print(result1)
#print('Precision Score: ')
#print(precision_score(y_test,y_pred, average='None')) #Parameter average='micro' calculates global precision/recall.
#print("Recall Score: ")
#print(recall_score(y_test,y_pred, average='None'))
#Explanation:
#https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score


#trial 4 - https://www.listendata.com/2017/12/k-nearest-neighbor-step-by-step-tutorial.html

