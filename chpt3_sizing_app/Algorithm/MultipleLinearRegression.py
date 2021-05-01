#From: https://datatofish.com/multiple-linear-regression-python/

import pandas as pd
import math
import numpy

from sklearn import linear_model
import statsmodels.api as sm
#1 = S, narrow, small, flat
#2 = M, average
#3 = L, Large, Wider, Round/Curvy

#The training data is the satisfied size of an item together with information of the user.
# THE TRAINING DATA WOULD ONLY CONTAIN SATISFIED SIZES (meaning clothing that fits)  
#The very first tests shows that adding multiple variables do not provide a more precise prediction. 
#Heigh and weight is sufficient. Im afraid that the other variables would deviate too much, since they are personal assumptions and not somewhat precise measurements. 
    #Everyone knows somewhat of their height and weight. People dont often think about their body shapes. 
    #Age would also deviate too much, since a person who is 70 can be just as tall and fat as a person who is 25. 
        #perhaps if we get enough data, we can group people into their respective age-groups? 

#TODO:
#Figure out a way to determine the variable-weight of each variables.
    #There must be a statistical way of doing this. 

#Training data:
# Male_Inputs = { 'Size': [4, 5, 4, 5, 5,3,1,2,4,1],
#                 'Age': [20, 20, 20, 20, 20,20,20,20,20,20],
#                 'Height': [178, 183, 189, 180, 180, 179.8, 166,173,172,172],
#                 'Weight': [70, 85, 81, 75, 80, 65.7,58,64,77,59],
#                 'Bmi': [22.8, 25.3, 22.6, 23.1, 24.7, 20.3, 21, 21.3, 26, 19.9],
#                 'TummyShape': [2,1,2,1,2,1,1,2,3,2],
#                 'HipShape':   [2,2,2,3,3,2,2,2,2,1],
#                 'ChestShape': [2,2,2,3,3,2,2,2,2,1]
#                 }



# names = ['Age', 'Height', 'Weight', 'Bmi', 'TummyShape', 'HipShape', 'ChestShape', 'Size']
#df = pd.DataFrame(Male_Inputs,columns=['Size','Age', 'Weight','Height', 'Bmi','TummyShape', 'HipShape', 'ChestShape' ])
# Assign colum names to the dataset
names = ['gender','age','height','weight','bmi','tummy','hip','breast','baselayersize','jeserysize','bibsize']
# Read dataset to pandas dataframe
df1 = pd.read_csv('realUsers.csv', names=names)

def calc_predict(item):
    X = df1[['gender','age','height','weight','bmi','tummy','hip','breast']] 
    Y = df1[item]
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    return regr

# with sklearn
# regr = linear_model.LinearRegression()
# regr.fit(X, Y)
# print(regr.score(X,Y))

# print('Intercept: \n', regr.intercept_)
# print('Coefficients: \n', regr.coef_)

# prediction with sklearn
# new prediction for a user:

filename = 'BotUsersWithoutSize.csv'
df2 = pd.read_csv(filename)

botUsersWithSize = {'gender':[],'age':[],'height':[],'weight':[], 'bmi':[],'tummy':[],'hip':[],'breast':[], 'baselayersize':[], 'jeserysize':[], 'bibsize':[]}

for index, row in df2.iterrows():
    gender          = int(row['gender'])
    age             = int(row['age'])
    height          = row['height']
    weight          = row['weight']
    bmi             = row['bmi']
    tummy           = int(row['tummy'])
    hip             = int(row['hip'])
    breast          = int(row['breast'])
    baselayersize   = round(numpy.float64(calc_predict('baselayersize').predict([[gender, age, height, weight, bmi, tummy, hip, breast]])))
    jeserysize      = round(numpy.float64(calc_predict('jeserysize').predict([[gender, age, height, weight, bmi, tummy, hip, breast]])))
    bibsize         = round(numpy.float64(calc_predict('bibsize').predict([[gender, age, height, weight, bmi, tummy, hip, breast]])))
    botUsersWithSize['gender'].append(gender)
    botUsersWithSize['age'].append(age)
    botUsersWithSize['height'].append(height)
    botUsersWithSize['weight'].append(weight)
    botUsersWithSize['bmi'].append(bmi)
    botUsersWithSize['tummy'].append(tummy)
    botUsersWithSize['hip'].append(hip)
    botUsersWithSize['breast'].append(breast)
    botUsersWithSize['baselayersize'].append(baselayersize)
    botUsersWithSize['jeserysize'].append(jeserysize)
    botUsersWithSize['bibsize'].append(bibsize)

print('bot users have been gathered and their size predicted for baselayer, jersey and bibs, see file: BotUsersWithSize.csv ')
df3 = pd.DataFrame(botUsersWithSize,columns=names)
df3.to_csv('BotUsersWithSize.csv', index=False)
#print ('Predicted Size: \n', regr.predict([[New_Age, New_Height, New_Weight, New_Bmi, New_TummyShape, New_HipShape, New_ChestShape]])) 


# with statsmodels
# X = sm.add_constant(X) # adding a constant

# model = sm.OLS(Y, X).fit()
# predictions = model.predict(X) 

# #Explains the model:
# #https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a

# print_model = model.summary()
# print(print_model)


