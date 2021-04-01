#From: https://datatofish.com/multiple-linear-regression-python/

import pandas as pd
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
Male_Inputs = { 'Size': [1, 5, 5, 4, 4],
                'Age': [20, 30, 35, 45, 25],
                'Height': [160, 175, 195, 182, 185],
                'Weight': [54, 87, 85, 73, 78],
                'TummyShape': [1,2,3,1,1],
                'HipShape': [1,2,3,2,2],
                'ChestShape': [1,2,3,2,2]
                }

Female_Inputs = { 'Size:': [],
                'Age': [],
                'Height': [],
                'Weight': [],
                'TummyShape': [],
                'HipShape': [],
                'BustShape': [],
                'ActualBraSize': []
                }

df = pd.DataFrame(Male_Inputs,columns=['Size','Age', 'Weight','Height','TummyShape', 'HipShape', 'ChestShape'
]) #,
X = df[[ 'Age', 'Height', 'Weight', 'TummyShape', 'HipShape', 'ChestShape']] #
Y = df['Size']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
# new prediction for a user:
New_Age = 25
New_Height = 170
New_Weight = 88
New_TummyShape = 2
New_HipShape = 1
New_ChestShape = 1
print ('Predicted Size: \n', regr.predict([[New_Age, New_Height, New_Weight, New_TummyShape, New_HipShape, New_ChestShape]])) 

# with statsmodels
X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)

