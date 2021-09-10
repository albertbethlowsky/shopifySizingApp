import csv
#import src.algo_nbc as nbc
import src.algo_knn as knn
import src.algo_dtc as dtc
import src.algo_svm as svm
import src.algo_svc as svc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

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

#_____________________WARMING THE MACHINE UP:____________________________________
# def run_warmup(X_test, y_test, X_train, y_train):
#     functionshapes = ['ovo', 'ovr']
#     for dfs in functionshapes:
#         svc.getresults(dfs, X_test, y_test, X_train, y_train)
#     print('WARMING UP IS DONE!')


#_____________________PLOTTING:____________________________________
def cmnPlot(algo, df, par1, par2, par3):
    fig = plt.figure()
    sns.heatmap(df, annot=True, fmt='.2%', cmap='Blues', annot_kws={'size': 15})
    plt.xlabel("Predicted") 
    plt.ylabel("Actual") 
    s = './plotting/CMN/' + algo + '/'+ par1 +'_'+par2+'_'+par3+'.png'
    fig.savefig(s)
    plt.clf()
    plt.close()

def cmPlot(algo, df, par1, par2, par3):
    fig = plt.figure()
    sns.heatmap(df, annot=True, cmap='Blues', annot_kws={'size': 15})
    plt.xlabel("Predicted") 
    plt.ylabel("Actual") 
    s = './plotting/CM/'+algo+'/' + par1 +'_'+par2+'_'+par3+'.png'
    fig.savefig(s)
    plt.clf()
    plt.close()

#_______________________Running the algorithm:________________________________
def run_test(name, X_test, y_test, X_train, y_train):
    #run_warmup(X_test, y_test, X_train, y_train)
    header_confusionmatrix = ['First Row', 'Second Row', 'Third Row']
    header_report = ['Fit', 'Precision', 'Recall', 'f1-score', 'support']
    filenameCSV = 'results/' + name + '.csv'
    
    #Column names: 
    l = ['name', 'subname', 'accuracy', 'rmse', 'mae', 'crossval_accuracy', 'crossval_accuracy_stnd_dev', 'timetotrain(fitting)']
    append_list_as_row_newfile(filenameCSV, l)
    #___________________________________________________
     #SUPPORT VECTOR MACHINE:
    functionshapes = ['ovo', 'ovr']
    for dfs in functionshapes:
        results=svc.getresults(dfs, X_test, y_test, X_train, y_train)
        #results:
        append_list_as_row(filenameCSV, results[0])
        #Classfication Report:
        append_list_as_row(filenameCSV, header_report)
        results[1].to_csv(filenameCSV, mode='a', header=False)
        #CM Plotting:
        #cmPlot('SVM', results[2], dfs, name, '')
        #CM:
        append_list_as_row(filenameCSV, header_confusionmatrix)
        results[2].to_csv(filenameCSV, mode='a', header=False)
        #CMN Plotting:
        cmnPlot('SVM', results[3], dfs, name, '')
        #CMN:
        append_list_as_row(filenameCSV, header_confusionmatrix)
        results[3].to_csv(filenameCSV, mode='a', header=False)

    print('Support Vector Model (One vs One) - done: ')
    #___________________________________________________
    #SUPPORT VECTOR MACHINE - Crammer Singer and One Vs Rest:
    schemes = ['crammer_singer', 'ovr'] #crammersinger VS one vs rest
    for scheme in schemes:
        results=svm.getresults(scheme, X_test, y_test, X_train, y_train)
        #results:
        append_list_as_row(filenameCSV, results[0])
        #Classfication Report:
        append_list_as_row(filenameCSV, header_report)
        results[1].to_csv(filenameCSV, mode='a', header=False)
        #CM Plotting:
        #cmPlot('SVM', results[2], scheme, name, '')
        #CM:
        append_list_as_row(filenameCSV, header_confusionmatrix)
        results[2].to_csv(filenameCSV, mode='a', header=False)
        #CMN Plotting:
        cmnPlot('SVM', results[3], scheme, name, '')
        #CMN:
        append_list_as_row(filenameCSV, header_confusionmatrix)
        results[3].to_csv(filenameCSV, mode='a', header=False)

    print('Support Vector Model (Crammer singer and OneVsRest) - done: ')
    #___________________________________________________
    #DECISION TREE CLASSIFIER:
    for depth in range(3,20):
        if(depth%2)==0:
            s = 'depth='+str(depth)
            #Results:
            results=dtc.getresults(depth, X_test, y_test, X_train, y_train)
            append_list_as_row(filenameCSV, results[0])
            #Classfication Report:
            append_list_as_row(filenameCSV, header_report)
            results[1].to_csv(filenameCSV, mode='a', header=False)
            #CM Plotting:
            #cmPlot('DTC', results[2], s, name, '')
            #CM:
            append_list_as_row(filenameCSV, header_confusionmatrix)
            results[2].to_csv(filenameCSV, mode='a', header=False)
            #CMN Plotting:
            cmnPlot('DTC', results[3], s, name, '')
            #CMN:
            append_list_as_row(filenameCSV, header_confusionmatrix)
            results[3].to_csv(filenameCSV, mode='a', header=False)
    print('Decision Tree Classifier - done')
    # #__________________________________________________
    # #NAIVE BAYES CLASSIFIER:
    # algos = ['Gaussian']
    # for algo in algos:
    #     #Results:
    #     results=nbc.getresults(algo, X_test, y_test, X_train, y_train)
    #     append_list_as_row(filenameCSV, results[0])
    #     #Classfication Report:
    #     append_list_as_row(filenameCSV, header_report)
    #     results[1].to_csv(filenameCSV, mode='a', header=False)
    #     #CM:
    #     append_list_as_row(filenameCSV, header_confusionmatrix)
    #     results[2].to_csv(filenameCSV, mode='a', header=False)
    #     #CMN:
    #     append_list_as_row(filenameCSV, header_confusionmatrix)
    #     results[3].to_csv(filenameCSV, mode='a', header=False)
    # print('Naive Bayes Classifier - done')
    # #_________________________________________________
    #K-NEAREST NEIGHBOR:
    weights = ['uniform', 'distance']
    for weight in weights:
        if(len(X_test)<1000):
            for k in range(1,11): #cross validation
                if (k % 2) != 0:  
                    w = 'W='+weight
                    s = 'K='+str(k)
                    results=knn.getresults(weight, k, X_test, y_test, X_train, y_train)
                    #results:
                    append_list_as_row(filenameCSV, results[0])
                    #Classfication Report:
                    append_list_as_row(filenameCSV, header_report)
                    results[1].to_csv(filenameCSV, mode='a', header=False)
                    #CM Plotting:
                    #cmPlot('KNN',results[2], w, s, name)
                    #CM:
                    append_list_as_row(filenameCSV, header_confusionmatrix)
                    results[2].to_csv(filenameCSV, mode='a', header=False)
                    #CMN Plotting:
                    cmnPlot('KNN', results[3], w, s, name)
                    #CMN:
                    append_list_as_row(filenameCSV, header_confusionmatrix)
                    results[3].to_csv(filenameCSV, mode='a', header=False)
        else:
            for k in range(10,125):
                if (k % 2) != 0:  
                    w = 'W='+weight
                    s = 'K='+str(k)
                    #results:
                    results=knn.getresults(weight, k, X_test, y_test, X_train, y_train)
                    append_list_as_row(filenameCSV, results[0])
                    #Classfication Report:
                    append_list_as_row(filenameCSV, header_report)
                    results[1].to_csv(filenameCSV, mode='a', header=False)
                    #CM Plotting:
                    #cmPlot('KNN', results[2], w, s, name)
                    #CM:
                    append_list_as_row(filenameCSV, header_confusionmatrix)
                    results[2].to_csv(filenameCSV, mode='a', header=False)
                    #CMN Plotting:
                    cmnPlot('KNN', results[3], w, s, name)
                    #CMN:
                    append_list_as_row(filenameCSV, header_confusionmatrix)
                    results[3].to_csv(filenameCSV, mode='a', header=False)
    print('K-Nearest Neighbors - done')
    print('_______________________________')
    print('DONE! - see file: ', filename)


#__________________________INPUT VARIABLES_________________________#
#clean runway data:
#import src.treat_RunWay as treat
#treat.create_csv('./Data/renttherunway_final_data.json', './Data/clean_runway.csv')
#print('done cleaning')

#  #_______________________TEST 1:________________________________
# #| Runway on Runway Trained Model, 20 % runway test set. |
# filename = 'Runway_M+F_20%'
# testsizerunway = 0.2

# #______DATA PREP START ________
# #runway:
# import src.dataprep_runway as runway
# l = runway.runwayvals(testsizerunway)
# X_train_runway = l[0]
# X_test_runway = l[1]
# y_train_runway = l[2]
# y_test_runway = l[3]
# #______DATA PREP END ________

# run_test(filename, X_test_runway, y_test_runway, X_train_runway, y_train_runway)
# print('-----TEST 1 DONE (runway_on_runway_maleandfemale_testsize20prct)------')

# #_______________________TEST 2:________________________________
# #| Chpt3 on Chpt3 Trained Model, male and female, 20 % chpt3 test set. |
# filename = 'Chpt3_F_20%'
# testsizechpt3 = 0.2
# genders = ['male'] #parse empty list if you want both male and female. Include gender to exclude that gender.

# #______DATA PREP START ________
# #chpt3:
# import src.dataprep_chpt3 as chpt3
# l = chpt3.chpt3vals(genders, testsizechpt3) 
# X_train_chpt3 = l[0]
# X_test_chpt3 = l[1]
# y_train_chpt3 = l[2]
# y_test_chpt3 = l[3]
# #______DATA PREP END ________

# run_test(filename, X_test_chpt3, y_test_chpt3, X_train_chpt3, y_train_chpt3)

# print('-----TEST 2 DONE (chpt3_on_chpt3_female_testsize20prct)------')


# #_______________________TEST 3:________________________________
# #| Chpt3 on Runway Trained Model, both male and female, 100 % |

# filename = 'Chpt3_Runway_M+F_100%'

# testsizerunway = 1
# testsizechpt3 = 1
# genders = [] #parse empty list if you want both male and female. Include gender to exclude that gender.

# #______DATA PREP START ________
# #chpt3:
# import src.dataprep_chpt3 as chpt3
# l = chpt3.chpt3vals(genders, testsizechpt3) 
# X_train_chpt3 = l[0]
# X_test_chpt3 = l[1]
# y_train_chpt3 = l[2]
# y_test_chpt3 = l[3]

# #runway:
# import src.dataprep_runway as runway
# l = runway.runwayvals(testsizerunway)
# X_train_runway = l[0]
# X_test_runway = l[1]
# y_train_runway = l[2]
# y_test_runway = l[3]
# #______DATA PREP END ________
 
# run_test(filename, X_test_chpt3, y_test_chpt3, X_train_runway, y_train_runway)
# print('-----TEST 3 DONE (chpt3_on_runway_maleandfemale_testsize100prct)------')

# #_______________________TEST 4:________________________________
# #| Chpt3 on Runway Trained Model, only female, 100 % chpt3 test set. |

# filename = 'Chpt3_Runway_F_100%'
# testsizerunway = 1
# testsizechpt3 = 1
# genders = ['male'] #parse empty list if you want both male and female. Include gender to exclude that gender.

# #______DATA PREP START ________
# #chpt3:
# import src.dataprep_chpt3 as chpt3
# l = chpt3.chpt3vals(genders, testsizechpt3) 
# X_train_chpt3 = l[0]
# X_test_chpt3 = l[1]
# y_train_chpt3 = l[2]
# y_test_chpt3 = l[3]

# #runway:
# import src.dataprep_runway as runway
# l = runway.runwayvals(testsizerunway)
# X_train_runway = l[0]
# X_test_runway = l[1]
# y_train_runway = l[2]
# y_test_runway = l[3]
# #______DATA PREP END ________

# run_test(filename, X_test_chpt3, y_test_chpt3, X_train_runway, y_train_runway)
# print('-----TEST 4 DONE (chpt3_on_runway_female_testsize100prct)------')

# #_______________________TEST 5:________________________________
# #| Chpt3 on Chpt3 Trained Model, male and female, 20 % chpt3 test set. |
# filename = 'Chpt3_M+F_20%'
# testsizechpt3 = 0.2
# genders = [] #parse empty list if you want both male and female. Include gender to exclude that gender.

# #______DATA PREP START ________
# #chpt3:
# import src.dataprep_chpt3 as chpt3
# l = chpt3.chpt3vals(genders, testsizechpt3) 
# X_train_chpt3 = l[0]
# X_test_chpt3 = l[1]
# y_train_chpt3 = l[2]
# y_test_chpt3 = l[3]
# #______DATA PREP END ________

# run_test(filename, X_test_chpt3, y_test_chpt3, X_train_chpt3, y_train_chpt3)



# print('-----TEST 5 DONE (chpt3_on_chpt3_maleandfemale_testsize20prct)------')


#_______________________TEST 6:________________________________
#| Single input on Runway Trained Model, 100 % single input test set. |
filename = 'Single_Runway_M+F_100%'
testsizerunway = 1

#______DATA PREP START ________
#runway:
import src.dataprep_runway_single as single
l = single.runwayvals(testsizerunway)
X_train_single = l[0]
X_test_single = l[1]
y_train_single = l[2]
y_test_single = l[3]

#runway:
import src.dataprep_runway as runway
l = runway.runwayvals(testsizerunway)
X_train_runway = l[0]
X_test_runway = l[1]
y_train_runway = l[2]
y_test_runway = l[3]
#______DATA PREP END ________

run_test(filename, X_test_single, y_test_single, X_train_runway, y_train_runway)

print('-----TEST 6 DONE (single on runway)------')



