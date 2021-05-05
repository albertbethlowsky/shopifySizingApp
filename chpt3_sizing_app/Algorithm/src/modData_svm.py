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
import src.treatModClothData as treatData
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
dataset = treatData.create_df()
#dataset.info()

#label the data, such that strings get an int representation
le.fit(dataset.quality)
quality_label = le.transform(dataset.quality)

le.fit(dataset.cup_size)
cup_size_label = le.transform(dataset.cup_size)

le.fit(dataset.hips)
hips_label = le.transform(dataset.hips)

le.fit(dataset.bra_size)
bra_size_label = le.transform(dataset.bra_size)

le.fit(dataset.category)
category_label = le.transform(dataset.category)

le.fit(dataset.length)
length_label = le.transform(dataset.length)

le.fit(dataset.fit)
fit_label = le.transform(dataset.fit)

le.fit(dataset.shoe_size)
shoe_size_label = le.transform(dataset.shoe_size)

le.fit(dataset.shoe_width)
shoe_width_label = le.transform(dataset.shoe_width)

dataset.quality = quality_label
dataset.cup_size = cup_size_label
dataset.hips = hips_label
dataset.bra_size = bra_size_label
dataset.category = category_label
dataset.length = length_label
dataset.fit = fit_label
dataset.shoe_size = shoe_size_label
dataset.shoe_width = shoe_width_label


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
uniquevalues = np.unique(dataset[['item_id']].values)
for id in uniquevalues:
    newDataset = dataset[dataset['item_id']==id]
    #newDataset = newDataset[newDataset['fit']==0]
    #print(newDataset.head(20))
    newDataset = newDataset[['size', 'cup_size','hips','bra_size','height','shoe_size','shoe_width','fit']].copy()
    if(len(newDataset)>1500):
        #print('new ID: ')
        #print(id)
        #plot_correlation(newDataset)
        #split dataset:
        X = newDataset.iloc[:,0:6]
        y = newDataset.iloc[:,7]
        # print(X)
        # print(y)

        # dividing X, y into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

        from sklearn.svm import SVC
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        for i in kernels:
            svm_model_linear = SVC(kernel = i, C = 1).fit(X_train, y_train)
            svm_predictions = svm_model_linear.predict(X_test)
            
            # model accuracy for X_test  
            accuracy = svm_model_linear.score(X_test, y_test)
            #print(accuracy)
            
            # creating a confusion matrix
            cm = confusion_matrix(y_test, svm_predictions)
            #print(cm)


            # print('Classification Report:')
            # print(classification_report(y_test, svm_predictions))
            # print('Root Mean Squared Error (RMSE):')
            # print(mean_squared_error(y_test,svm_predictions))
            # print('Mean Aboslute Error (MAE):')
            # print(mean_absolute_error(y_test,svm_predictions))
            # print('_______________________________________')
            
            # s = 'Support Vector Machine,'+i+','+str(accuracy)+','+str(mean_squared_error(y_test,svm_predictions))+','+str(mean_absolute_error(y_test,svm_predictions))
            # print(s)
            users = len(newDataset)
            import src.appendToCsv as atc
            row_contents = ['Support Vector Machine',i,users,id,accuracy,mean_squared_error(y_test,svm_predictions), mean_absolute_error(y_test,svm_predictions)]
            atc.append_list_as_row('results/MLresults_modData.csv', row_contents)

