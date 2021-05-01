#From: https://datatofish.com/multiple-linear-regression-python/

import pandas as pd
import math
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
Male_Inputs = { 'Size': [4, 5, 4, 5, 5,3,1,2,4,1],
                'Age': [20, 20, 20, 20, 20,20,20,20,20,20],
                'Height': [178, 183, 189, 180, 180, 179.8, 166,173,172,172],
                'Weight': [70, 85, 81, 75, 80, 65.7,58,64,77,59],
                'Bmi': [22.8, 25.3, 22.6, 23.1, 24.7, 20.3, 21, 21.3, 26, 19.9],
                'TummyShape': [2,1,2,1,2,1,1,2,3,2],
                'HipShape':   [2,2,2,3,3,2,2,2,2,1],
                'ChestShape': [2,2,2,3,3,2,2,2,2,1]
                }

Female_Inputs = { 'Size:': [],
                'Age': [],
                'Height': [],
                'Weight': [],
                'Bmi': [],
                'TummyShape': [],
                'HipShape': [],
                'BustShape': [],
                'ActualBraSize': []
                }

names = ['Age', 'Height', 'Weight', 'Bmi', 'TummyShape', 'HipShape', 'ChestShape', 'Size']

#df = pd.read_csv('usersnew.csv', names=names)

df = pd.DataFrame(Male_Inputs,columns=['Size','Age', 'Weight','Height', 'Bmi','TummyShape', 'HipShape', 'ChestShape' ])
X = df[[ 'Age', 'Height', 'Weight', 'Bmi', 'TummyShape', 'HipShape', 'ChestShape']] 
Y = df['Size']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print(regr.score(X,Y))

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
# new prediction for a user:
New_Age = 35
New_Height = 190
New_Weight = 100

New_Bmi = New_Weight / pow(New_Height*0.01,2)
print(New_Bmi)
New_TummyShape = 3
New_HipShape = 3
New_ChestShape = 3
print ('Predicted Size: \n', regr.predict([[New_Age, New_Height, New_Weight, New_Bmi, New_TummyShape, New_HipShape, New_ChestShape]])) 

# with statsmodels
X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

#Explains the model:
#https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a

print_model = model.summary()
print(print_model)


