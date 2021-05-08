#SVM (Support vector machine) classifier 
# importing necessary libraries
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
  
def run(featureMin, featureMax, label):
    dataset = pd.read_csv('./src/botUsersWithSize.csv')
    # loading the iris dataset
    iris = datasets.load_iris()
    
    # X -> features, y -> label
    X = dataset.iloc[:,featureMin:featureMax] #X = iris.data
    y = dataset.iloc[:,label] #y = iris.target

    
    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    # training a linear SVM classifier
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


        print('Classification Report for SVM:')
        print(classification_report(y_test, svm_predictions, zero_division=1))
        # print('Root Mean Squared Error (RMSE):')
        # print(mean_squared_error(y_test,svm_predictions))
        # print('Mean Aboslute Error (MAE):')
        # print(mean_absolute_error(y_test,svm_predictions))
        # print('_______________________________________')
        
        # s = 'Support Vector Machine,'+i+','+str(accuracy)+','+str(mean_squared_error(y_test,svm_predictions))+','+str(mean_absolute_error(y_test,svm_predictions))
        # print(s)
        import src.appendToCsv as atc
        filename = 'results/MLresults_ownData_label#' + str(label) + '.csv'
        row_contents = ['Support Vector Machine',i,accuracy,mean_squared_error(y_test,svm_predictions), mean_absolute_error(y_test,svm_predictions)]
        atc.append_list_as_row(filename, row_contents)
