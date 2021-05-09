import csv
import src.appendToCsv as atc

l = ['name', 'subname', 'n','item_id','accuracy', 'rmse', 'mae']
atc.append_list_as_row_newfile('results/MLresults_modData.csv', l)

exec(open("./src/modData_svm.py").read())
print('Support Vector Model - done')
exec(open("./src/modData_knn.py").read())
print('K-Nearest Neighbors - done')
exec(open("./src/modData_nbc.py").read())
print('Naive Bayes Classifier - done')
exec(open("./src/modData_dtc.py").read())
print('Decision Tree Classifier - done')
print('_______________________________')
print('DONE! - see file: results/MLresults_modData.csv')