import csv
import src.appendToCsv as atc

import src.generateUsers as gu
#gu.generate(10000)

l = ['name', 'subname', 'accuracy', 'rmse', 'mae']
atc.append_list_as_row_newfile('results/MLresults_ownData.csv', l)

exec(open("./src/ownData_svm.py").read())
print('Support Vector Model - done')
exec(open("./src/ownData_knn.py").read())
print('K-Nearest Neighbors - done')
exec(open("./src/ownData_nbc.py").read())
print('Naive Bayes Classifier - done')
exec(open("./src/ownData_dtc.py").read())
print('Decision Tree Classifier - done')
print('_______________________________')

fileObject = csv.reader('/src/BotUsersWithSize.csv')
row_count = sum(1 for row in fileObject)  # fileObject is your csv.reader

with open('./src/BotUsersWithSize.csv') as f:
    s = sum(1 for line in f)
    print('Nr of users tested:')
    print(s-1)

print('DONE! - see file: results/MLresults_ownData.csv')
