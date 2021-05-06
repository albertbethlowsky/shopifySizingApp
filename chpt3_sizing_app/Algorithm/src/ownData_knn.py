import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
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
import src.plotting as plotting

def run(featureMin, featureMax, label):
    #make bot users and have them assigned sizes with the help of multipleLinearRegression and actual data. 

    #Create new csv of users


    # Read dataset to pandas dataframe
    dataset = pd.read_csv('./src/botUsersWithSize.csv')


    # # #print(len(dataset.index)) #number of items
    # # X = dataset.iloc[:, :-1].values #size
    # # y = dataset.iloc[:, 6].values #Age, height, weight, tummyshape, hipshape, chestshape

    # # from sklearn.model_selection import train_test_split
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # # from sklearn.preprocessing import StandardScaler
    # # scaler = StandardScaler()
    # # scaler.fit(X_train)

    # # X_train = scaler.transform(X_train)
    # # X_test = scaler.transform(X_test)

    # # from sklearn.neighbors import KNeighborsClassifier
    # # classifier = KNeighborsClassifier(n_neighbors=3) #len(dataset.index) = number of rows in dataset
    # # classifier.fit(X_train, y_train)

    # # y_pred = classifier.predict(X_test)

    # # from sklearn.metrics import classification_report, confusion_matrix
    # # print(confusion_matrix(y_test, y_pred))
    # # print(classification_report(y_test, y_pred))



    # #new try:
    # #https://medium.com/codex/machine-learning-k-nearest-neighbors-algorithm-with-python-df94b374ad41

    style.use('seaborn-whitegrid')
    plt.rcParams['figure.figsize'] = (16, 7)

    # #Scatter plot:
    # sb.scatterplot('Height', 'Weight', data = dataset, hue = 'Size', palette = 'Set2', edgecolor = 'b', s = 150, 
    #                alpha = 0.7)
    # plt.title('Height / Weight')
    # plt.xlabel('Height')
    # plt.ylabel('Weight')
    # plt.legend(loc = 'upper left', fontsize = 12)
    # plt.savefig('heightweight.png')

    # # 4. Scatter Matrix:
    #cols_to_plot = dataset.columns[1:8].tolist() 
    #sb.pairplot(data = dataset, hue = 'baselayersize')
    #plt.savefig('pairplot1.png')

    # #Training the data
    # X_var = dataset[['Height', 'Weight']].values
    # #print(X_var)
    # y_var = dataset['Size'].values

    # print(cl('X variable :', attrs = ['bold']), X_var[:5])
    # print(cl('Y variable :', attrs = ['bold']), y_var[:5])

    # #Normalize:
    # X_var = StandardScaler().fit(X_var).transform(X_var.astype(float))
    # #print(cl(X_var[:5], attrs = ['bold'])) #see dependent and independent variables?

    # #train:
    # X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 0)

    # print(cl('Train set shape :', attrs = ['bold']), X_train.shape, y_train.shape)
    # print(cl('Test set shape :', attrs = ['bold']), X_test.shape, y_test.shape)

    # #knn algo:
    # k = int(math.sqrt(len(dataset.index)))#sqrt of n, seems to be good k value. k must be uneven, such that the result wont be even between two clusters 

    # if (k % 2) == 0:  
    #    k = k-1 #if k is even, subtract one. 
    # print('this is k: ')
    # print(k)

    # neigh = KNeighborsClassifier(n_neighbors = k)
    # neigh.fit(X_train, y_train)

    # print(cl(neigh, attrs = ['bold']))

    # #prediction
    # yhat = neigh.predict(X_test)

    # print(cl('Prediction Accuracy Score (%) :', attrs = ['bold']), round(accuracy_score(y_test, yhat)*100, 2))

    # # input candidate
    # # names = ['Height', 'Weight']

    # # dataset = pd.read_csv('testuser.csv', names=names)

    # # testUser = dataset[['Height', 'Weight']].values

    # # testUserPredict = neigh.predict(testUser)
    

    # # print(cl('Prediction Accuracy Score (%) :', attrs = ['bold']), round(accuracy_score(testUser, testUserPredict)*100, 2))
    # #print(arrOfArr)
    # # newPrediction = { 'Age': [25],
    # #                 'Height': [170],
    # #                 'Weight': [88],
    # #                 'TummyShape': [2],
    # #                 'HipShape': [1],
    # #                 'ChestShape': [1]
    # #                 }

    # #df = pd.DataFrame(arrOfArr,columns=['Age', 'Weight','Height','TummyShape', 'HipShape', 'ChestShape']) 




    # #print(cl('Prediction Accuracy Score (%) :', attrs = ['bold']), round(accuracy_score(newPrediction, yhat)*100, 2))


    #Trial 3 - https://www.youtube.com/watch?v=4HKqjENq9OU


    #KNN - predict whether a person is xs,s,m,l or xl

    #replace zero's (clean data)
    # zero_not_accepted = ['gender','age','height','weight','bmi','tummy','hip', 'breast']
    # for column in zero_not_accepted:
    #     dataset[column] = dataset[column].replace(0, np.NaN)
    #     mean = int(dataset[column].mean(skipna=True))
    #     dataset[column] = dataset[column].replace(np.NaN, mean)

    

    #plot_correlation(dataset)

    #split dataset:
    X = dataset.iloc[:,featureMin:featureMax]
    y = dataset.iloc[:,label]
    print(X)
    print(y)

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

    s = 'k= '+str(k)
    import src.appendToCsv as atc
    filename = 'results/MLresults_ownData_label#' + str(label) + '.csv'
    row_contents = ['K-Nearest Neighbor',s,accuracy_score(y_test, y_pred),mean_squared_error(y_test,y_pred), mean_absolute_error(y_test,y_pred)]
    atc.append_list_as_row(filename, row_contents)

    #testing various uneven k, uneven is important for multi class
    # for i in range(3,15,2):
    #     knn = KNeighborsClassifier(n_neighbors=i, p=2, metric='euclidean')
    #     #Train the model using the training sets
    #     knn.fit(X_train, y_train)
    #     #Predict the response for test dataset
    #     y_pred = knn.predict(X_test)
    #     print('Accuracy:',accuracy_score(y_test, y_pred), 'for ',i)
    #     #accuracy_list.append(accuracy_score(y_test, y_pred))


    # #evaluate model
    cm = confusion_matrix(y_test, y_pred)
    # print (cm)

    # print('Root Mean Squared Error (RMSE):')
    # print(mean_squared_error(y_test,y_pred))
    # print('Mean Aboslute Error (MAE):')
    # print(mean_absolute_error(y_test,y_pred))
    # print('Accuracy Score:')
    # # print(f1_score(y_test,y_pred))
    # print(accuracy_score(y_test, y_pred))
    # print('Precision Score: ')
    # print(precision_score(y_test,y_pred, average='None')) #Parameter average='micro' calculates global precision/recall.
    # print("Recall Score: ")
    result1 = classification_report(y_test, y_pred, zero_division=1)
    print('Classification Report for KNN:')
    print(result1)

    #trial 4 - https://www.listendata.com/2017/12/k-nearest-neighbor-step-by-step-tutorial.html



def plot_correlation(data):
        '''
        plot correlation's matrix to explore dependency between features 
        '''
        # init figure size
        rcParams['figure.figsize'] = 15, 20
        fig = plt.figure()
        sns.heatmap(data.corr(), annot=True, fmt=".2f")
        plt.show()
        fig.savefig('corr.png')