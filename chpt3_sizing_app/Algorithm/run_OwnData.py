import csv
import src.appendToCsv as atc
import src.ownData_nbc as nbc
import src.ownData_knn as knn
import src.ownData_dtc as dtc
import src.ownData_svm as svm

#__________________________INPUT VARIABLES_________________________#
#To generate new users: 
import src.generateUsers as gu
nrOfUsers = 10000
#gu.generate(nrOfUsers)

#CHOOSE FEATURES AND LABELS:
#0=gender, 1=age, 2=height, 3=weght, 4=bmi, 5=tummy, 6=hip, 7=breast
#choose featuresMax=8 to include 7=breast
featuresMin = 0
featuresMax = 8
label = 8 #8=baselayersize, 9=jerseysize, 10=bibsize
#___________________________________________________________________#

filename = 'results/MLresults_ownData_label#' + str(label) + '.csv'
l = ['name', 'subname', 'accuracy', 'rmse', 'mae']
atc.append_list_as_row_newfile(filename, l)

svm.run(featuresMin,featuresMax,label)
print('Support Vector Model - done')
knn.run(featuresMin,featuresMax,label)
print('K-Nearest Neighbors - done')
nbc.run(featuresMin,featuresMax,label)
print('Naive Bayes Classifier - done')
dtc.run(featuresMin,featuresMax,label)
print('Decision Tree Classifier - done')
print('_______________________________')

fileObject = csv.reader('/src/BotUsersWithSize.csv')
row_count = sum(1 for row in fileObject)  # fileObject is your csv.reader

with open('./src/BotUsersWithSize.csv') as f:
    s = sum(1 for line in f)
    print('Nr of users tested:')
    print(s-1)

print('DONE! - see file: ', filename)
