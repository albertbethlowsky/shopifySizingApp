import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assign colum names to the dataset
names = ['Age', 'Height', 'Weight', 'TummyShape', 'HipShape', 'ChestShape', 'Size']

# Read dataset to pandas dataframe
dataset = pd.read_csv('users.csv', names=names)


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

import pandas as pd # data processing
from termcolor import colored as cl # elegant printing of text
import seaborn as sb # visualizations
import matplotlib.pyplot as plt # editing visualizations
import math
from matplotlib import style # setting styles for plots
from sklearn.preprocessing import StandardScaler # normalizing data
from sklearn.neighbors import KNeighborsClassifier # KNN algorithm
from sklearn.metrics import accuracy_score # algorithm accuracy
from sklearn.model_selection import train_test_split # splitting the data

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

sb.pairplot(data = dataset, hue = 'Size', palette = ['Red', 'Blue', 'limegreen', 'Orange', 'Green'])
plt.savefig('pairplot1.png')

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
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split # splits the data, part is training and part is testing
from sklearn.preprocessing import StandardScaler # So we dont have bias of high numbers. Uniformed of -1 and 1
from sklearn.neighbors import KNeighborsClassifier #the actual tool
#three tools to test:
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#KNN - predict whether a person is xs,s,m,l or xl

names = ['Age', 'Height', 'Weight', 'TummyShape', 'HipShape', 'ChestShape', 'Size']
dataset = pd.read_csv('users.csv', names=names)

#replace zero's (clean data)
zero_not_accepted = ['Age','Height','Weight','TummyShape','HipShape','ChestShape']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

#split dataset:
X = dataset.iloc[:,0:6]
y = dataset.iloc[:,6]

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
print (cm)
#print(f1_score(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

#trial 4 - https://www.listendata.com/2017/12/k-nearest-neighbor-step-by-step-tutorial.html

