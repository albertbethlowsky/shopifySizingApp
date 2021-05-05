#From: https://datatofish.com/multiple-linear-regression-python/

import pandas as pd
import math
import numpy
import random


from sklearn import linear_model
import statsmodels.api as sm

df1 = pd.read_csv('realUsers.csv')

#predict for one user:
def calc_predict(item):
    X = df1[['gender','age','height','weight','bmi','tummy','hip','breast']] 
    Y = df1[item]
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    return regr


def makeUsers(nrOfUsers):
    user = {'gender':[],'age':[],'height':[],'weight':[], 'bmi':[],'tummy':[],'hip':[],'breast':[]}

    #BMI scale: https://www.euro.who.int/en/health-topics/disease-prevention/nutrition/a-healthy-lifestyle/body-mass-index-bmi
    #Below 18.5: underweight
    #18.5-24.9: normal weight
    #25.0-29.9: pre-obesity
    #30.0-34.9: obesity class 1 
    #.. etc, but relevant? i think not. 

    male = 1
    female = 2
    ageMin = 18
    ageMax = 75
    heightMin = 145
    heightMax = 211
    weightMin = 41
    weightMax = 132
    usersProduced = 0

    while(usersProduced!=nrOfUsers):
        gender = random.randint(male,female)
        age = random.randint(ageMin,ageMax) 
        height = random.randint(heightMin,heightMax)
        weight = random.randint(weightMin, weightMax)
        bmi = round(weight / pow(height*0.01,2), 1)

        if(bmi>24.9 and bmi<32.5): #overweight
            tummy = random.randint(2,3)
            hip = random.randint(2,3)
            breast = random.randint(2,3)
            user['gender'].append(gender)
            user['age'].append(age)
            user['height'].append(height)
            user['weight'].append(weight)
            user['bmi'].append(bmi)
            user['tummy'].append(tummy)
            user['hip'].append(hip)
            user['breast'].append(breast)
            usersProduced+=1
        elif(bmi<18.5 and bmi>15): #underweight
            tummy = random.randint(1,2)
            hip = random.randint(1,2)
            breast = random.randint(1,2)
            user['gender'].append(gender)
            user['age'].append(age)
            user['height'].append(height)
            user['weight'].append(weight)
            user['bmi'].append(bmi)
            user['tummy'].append(tummy)
            user['hip'].append(hip)
            user['breast'].append(breast)
            usersProduced+=1
        elif(bmi>18.5 and bmi<25): #normalweight
            tummy = random.randint(1,3)
            hip = random.randint(1,3)
            breast = random.randint(1,3)
            user['gender'].append(gender)
            user['age'].append(age)
            user['height'].append(height)
            user['weight'].append(weight)
            user['bmi'].append(bmi)
            user['tummy'].append(tummy)
            user['hip'].append(hip)
            user['breast'].append(breast)
            usersProduced+=1

    df = pd.DataFrame(user,columns=['gender','age','height','weight','bmi','tummy','hip','breast'])
    df.to_csv('BotUsersWithoutSize.csv', index=False)

#generate all users
def generate(nrOfUsers):
    makeUsers(nrOfUsers)
    filename = 'BotUsersWithoutSize.csv'
    
    df2 = pd.read_csv(filename)

    botUsersWithSize = {'gender':[],'age':[],'height':[],'weight':[], 'bmi':[],'tummy':[],'hip':[],'breast':[], 'baselayersize':[], 'jerseysize':[], 'bibsize':[]}

    #including real users:
    for index, row in df1.iterrows():
        gender          = int(row['gender'])
        age             = int(row['age'])
        height          = row['height']
        weight          = row['weight']
        bmi             = row['bmi']
        tummy           = int(row['tummy'])
        hip             = int(row['hip'])
        breast          = int(row['breast'])
        baselayersize   = int(row['baselayersize'])
        jerseysize      = int(row['jerseysize'])
        bibsize         = int(row['bibsize'])
        botUsersWithSize['gender'].append(gender)
        botUsersWithSize['age'].append(age)
        botUsersWithSize['height'].append(height)
        botUsersWithSize['weight'].append(weight)
        botUsersWithSize['bmi'].append(bmi)
        botUsersWithSize['tummy'].append(tummy)
        botUsersWithSize['hip'].append(hip)
        botUsersWithSize['breast'].append(breast)
        botUsersWithSize['baselayersize'].append(baselayersize)
        botUsersWithSize['jerseysize'].append(jerseysize)
        botUsersWithSize['bibsize'].append(bibsize)
    #including bots:
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
        jerseysize      = round(numpy.float64(calc_predict('jerseysize').predict([[gender, age, height, weight, bmi, tummy, hip, breast]])))
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
        botUsersWithSize['jerseysize'].append(jerseysize)
        botUsersWithSize['bibsize'].append(bibsize)

    print('bot users have been gathered and their size predicted for baselayer, jersey and bibs, see file: BotUsersWithSize.csv ')
    df3 = pd.DataFrame(botUsersWithSize,columns=['gender','age','height','weight','bmi','tummy','hip','breast','baselayersize','jerseysize','bibsize'])
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


