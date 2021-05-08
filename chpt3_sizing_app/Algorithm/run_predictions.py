import csv
import src.algo_nbc as nbc
import src.algo_knn as knn
import src.algo_dtc as dtc
import src.algo_svm as svm

from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def append_list_as_row_newfile(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'w', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

#__________________________INPUT VARIABLES_________________________#
#clean runway data:
import src.treat_RunWay as treat
#treat.create_csv('C:/Users/Frederik/Desktop/shopifySizingApp/chpt3_sizing_app/Algorithm/Data/renttherunway_final_data.json', 'C:/Users/Frederik/Desktop/shopifySizingApp/chpt3_sizing_app/Algorithm/Data/clean_runway.csv')
#print('done cleaning')

testset = 'chpt3_on_runway_female_testsize100prct'
testsize = 0.2
genders = ['male'] #parse empty list if you want both male and female. Include gender to exclude that gender. 

#___________________________________________________________________#

filename = 'results/' + testset + '.csv'

#Column names: 
l = ['name', 'subname', 'accuracy', 'rmse', 'mae', 'crossval_accuracy', 'crossval_accuracy_stnd_dev']
append_list_as_row_newfile(filename, l)
#___________________________________________________________________#

#chpt3:
import src.dataprep_chpt3 as chpt3
l = chpt3.chpt3vals(genders, testsize) 
X_train_chpt3 = l[0]
X_test_chpt3 = l[1]
y_train_chpt3 = l[2]
y_test_chpt3 = l[3]

#runway:
import src.dataprep_runway as runway
l = runway.runwayvals(testsize)
X_train_runway = l[0]
X_test_runway = l[1]
y_train_runway = l[2]
y_test_runway = l[3]




#Support Vector Machine:
kernels = ['linear'] #'poly', 'rbf', 'sigmoid'
for i in kernels:
    results=svm.getresults(i, X_train_chpt3, X_test_chpt3, y_train_chpt3, y_test_chpt3, X_train_runway, X_test_runway, y_train_runway, y_test_runway)
    append_list_as_row(filename, results)
print('Support Vector Model - done')

#Decision Tree Classifier:
results=dtc.getresults(X_train_chpt3, X_test_chpt3, y_train_chpt3, y_test_chpt3, X_train_runway, X_test_runway, y_train_runway, y_test_runway)
append_list_as_row(filename, results)
print('Decision Tree Classifier - done')
print('_______________________________')

#K-Nearest Neighbor:
for k in range(1,20): #cross validation
    if (k % 2) != 0:  
        results=knn.getresults(k, X_train_chpt3, X_test_chpt3, y_train_chpt3, y_test_chpt3, X_train_runway, X_test_runway, y_train_runway, y_test_runway)
        append_list_as_row(filename, results)
        print('is done - knn nr: ', k)
print('K-Nearest Neighbors - done')

#Naive Bayes Classifier:
results=nbc.getresults(X_train_chpt3, X_test_chpt3, y_train_chpt3, y_test_chpt3, X_train_runway, X_test_runway, y_train_runway, y_test_runway)
append_list_as_row(filename, results)
print('Naive Bayes Classifier - done')





print('DONE! - see file: ', filename)





