import random
import pandas as pd
import csv
#Random is not representative, since it does not indicate likelihood. e.g. the majority of young people are more fit than elder. 
user = {'age':[],'height':[],'weight':[],'tummy':[],'hip':[],'breast':[], 'size':[]}
#Small, Fat person: 
for x in range(100):
    user['age'].append(random.randint(18,85))
    user['height'].append(random.randint(145,170))
    user['weight'].append(random.randint(80, 120))
    user['tummy'].append(random.randint(2,3))
    user['hip'].append(random.randint(2,3))
    user['breast'].append(random.randint(2,3))
    user['size'].append(random.randint(4,5))


#Small, Fit person: 
for x in range(100):
    user['age'].append(random.randint(18,85))
    user['height'].append(random.randint(145,170))
    user['weight'].append(random.randint(60, 75))
    user['tummy'].append(random.randint(1,3))
    user['hip'].append(random.randint(1,3))
    user['breast'].append(random.randint(1,3))
    user['size'].append(random.randint(2,3))

#small, thin person
for x in range(100):
    user['age'].append(random.randint(18,85))
    user['height'].append(random.randint(145,170))
    user['weight'].append(random.randint(45, 65))
    user['tummy'].append(random.randint(1, 2))
    user['hip'].append(random.randint(1, 2))
    user['breast'].append(random.randint(1,2))
    user['size'].append(random.randint(1,2))

#Tall, Fat person: 
for x in range(100):
    user['age'].append(random.randint(18,85))
    user['height'].append(random.randint(170,210))
    user['weight'].append(random.randint(90, 140))
    user['tummy'].append(random.randint(2,3))
    user['hip'].append(random.randint(2,3))
    user['breast'].append(random.randint(2,3))
    user['size'].append(random.randint(4,5))

#Tall, fit person: 
for x in range(100):
    user['age'].append(random.randint(18,85))
    user['height'].append(random.randint(170,210))
    user['weight'].append(random.randint(70, 95))
    user['tummy'].append(random.randint(1,3))
    user['hip'].append(random.randint(1,3))
    user['breast'].append(random.randint(1,3))
    user['size'].append(random.randint(2,3))

#Tall, thin person: 
for x in range(100):
    user['age'].append(random.randint(18,85))
    user['height'].append(random.randint(170,210))
    user['weight'].append(random.randint(65, 80))
    user['tummy'].append(random.randint(1,2))
    user['hip'].append(random.randint(1,2))
    user['breast'].append(random.randint(1,2))
    user['size'].append(random.randint(1,2))


df = pd.DataFrame(user,columns=['age','height','weight','tummy','hip','breast', 'size'
])
df.to_csv('usersnew.csv', index=False)

