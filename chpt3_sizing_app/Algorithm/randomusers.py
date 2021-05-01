import random
import pandas as pd
import csv
#These fictive users will be used on the model based method, to predict their sizes. The model based method is based on a small sample of users. 
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

while(usersProduced!=1000):
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
df.to_csv('usersnew.csv', index=False)

