
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import json
import re

#helper functions
def parse_bust_size(s):
    m = re.match(r'(\d+)([A-Za-z])(\+?)', s)
    if m:
        return pd.Series(data=[int(m.group(1)), m.group(2).lower()])
    return []

#there are 58 sizes in total, but we want to put them in to 8 categories
def parse_product_size(input):
    size = str(input)
    xxs = ['1','2','3','4','5'] 
    xs = ['6','7','8','9','10','11','12'] 
    s = ['13','14','15','16','17','18','19'] 
    m = ['20','21','22','23','24','25','26'] 
    l = ['27','28','29','30','31','32','33'] 
    xl = ['34','35','36','37','38','39','40'] 
    xxl = ['41','42','43','44','45','46'] 
    xxxl = ['47','48','49','50','51','52'] 
    xxxxl = ['53','54','55','56','57','58']
    if any(size in index for index in xxs):
        return pd.Series(data=['xxs'])
    elif any(size in index for index in xs):
        return pd.Series(data=['xs'])
    elif any(size in index for index in s):
        return pd.Series(data=['s'])
    elif any(size in index for index in m):
        return pd.Series(data=['m'])
    elif any(size in index for index in l):
        return pd.Series(data=['l'])
    elif any(size in index for index in xl):
        return pd.Series(data=['xl'])
    elif any(size in index for index in xxl):
        return pd.Series(data=['xxl'])
    elif any(size in index for index in xxxl):
        return pd.Series(data=['xxxl'])
    elif any(size in index for index in xxxxl):
        return pd.Series(data=['xxxxl'])
    # if (size in xxs):
    #     return pd.Series(data=[6]) 
    # elif(size in xs):
    #     return pd.Series(data=[4])
    # elif(size in s):
    #     return pd.Series(data=[2])
    # elif(size in m):
    #     return pd.Series(data=[1])
    # elif(size in l):
    #     return pd.Series(data=[0])
    # elif(size in xl):
    #     return pd.Series(data=[3])
    # elif(size in xxl):
    #     return pd.Series(data=[5])
    # elif(size in xxxl):
    #     return pd.Series(data=[7])

  

def convert_bust_size(input):
    s = str(input)
    if s in '28':
        return pd.Series(data=['60'])
    elif s in '30':
        return pd.Series(data=['65'])
    elif s in '32':
        return pd.Series(data=['70'])
    elif s in '34':
        return pd.Series(data=['75'])
    elif s in '36':
        return pd.Series(data=['80'])
    elif s in '38':
        return pd.Series(data=['85'])
    elif s in '40':
        return pd.Series(data=['90'])
    elif s in '42':
        return pd.Series(data=['95'])
    elif s in '44':
        return pd.Series(data=['100'])
    elif s in '46':
        return pd.Series(data=['105'])
    elif s in '48':
        return pd.Series(data=['110'])
    else:
        return pd.Series(data=['0'])

def parse_product_category(input):
    s = str(input)
    tops = ['blazer', 'blouse', 'blouson', 'bomber', 'buttondown','cami', 'cardigan', 'coat', 'crewneck', 'henley', 'hoodie', 'jacket', 'overcoat', 'parka', 'peacoat', 'pullover', 'shirt', 'sweater', 'sweater', 'sweatershirt', 'sweatshirt', 'tank', 'tee', 'tops', 'trench', 't-shirt', 'turtleneck', 'vest']
    bottoms = ['culotte', 'culottes', 'jeans', 'jogger', 'legging', 'leggings', 'overalls', 'pant', 'pants', 'skirt', 'skirts', 'skort', 'sweatpants', 'trouser', 'trousers']
    other = ['caftan', 'cape', 'combo', 'dress', 'duster', 'for', 'frock', 'gown', 'jumpsuit', 'kaftan', 'kimono', 'knit', 'maxi', 'midi','mini', 'poncho', 'print', 'romper', 'sheath', 'shift', 'shirtdress', 'suit', 'tight', 'tunic'] #not interested. 
    if any(s in index for index in tops):
        return pd.Series(data=['top'])
    elif any(s in index for index in bottoms):
        return pd.Series(data=['bottom'])
    elif any(s in index for index in other):
        return pd.Series(data=['other'])
    else:
        return pd.Series(data=['other'])

def load_data(fp):
    with open(fp) as fid:
        series = (pd.Series(json.loads(s)) for s in fid)
        return pd.concat(series, axis=1).T

def feet_to_meters(s):
    feet, inches = map(int, s.replace("'", '').replace("\"", '').split())
    return feet * 0.3048 + inches * 0.0254

def pounds_to_kilos(s):
    return int(s.replace('lbs', '')) * 0.45359237

#callable function:
def create_csv(fileinput, fileoutput):
    df = load_data(fileinput)
    to_drop = df[df['fit'] == 'fit'].isnull().any(axis=1)
    n = to_drop.sum()
    to_drop.shape, df.shape
    df = df.drop(df[df['fit'] == 'fit'][to_drop].index, axis=0)

    cleaned_df = df.copy()

    
    cleaned_df['height'] = (cleaned_df['height']
                            .fillna("0' 0\"")
                            .apply(feet_to_meters))
    cleaned_df['height'][cleaned_df['height'] == 0] = cleaned_df['height'].median()

    cleaned_df['weight'] = (cleaned_df['weight']
                        .fillna('0lbs')
                        .apply(pounds_to_kilos))
    cleaned_df['weight'][cleaned_df['weight'] == 0.0] = cleaned_df['weight'].median()


                        
    np.random.seed(69)
    fit_samples = cleaned_df[(cleaned_df['fit'] == 'fit')]
    selected_indices = np.random.choice(fit_samples.index, fit_samples.shape[0] // 2)
    fit_samples = fit_samples.loc[selected_indices]
    weight_shift_kg = 10.0

    augm_small_samples = fit_samples.copy()
    augm_small_samples['fit'] = 'small'
    augm_small_samples['weight'] -= weight_shift_kg

    augm_large_samples = fit_samples.copy()
    augm_large_samples['fit'] = 'large'
    augm_large_samples['weight'] += weight_shift_kg


    cleaned_df = cleaned_df.append(pd.concat((augm_large_samples, augm_small_samples)), ignore_index=True).dropna()

    cleaned_df['user_id'] = pd.to_numeric(cleaned_df['user_id'])
    cleaned_df['bust size'] = cleaned_df['bust size'].fillna(cleaned_df['bust size'].value_counts().index[0])
    cleaned_df['body type'] = cleaned_df['body type'].fillna(cleaned_df['body type'].value_counts().index[0])
    cleaned_df['item_id'] = pd.to_numeric(cleaned_df['item_id'])
    print(cleaned_df.describe())
    cleaned_df['size'] = pd.to_numeric(cleaned_df['size'])
    print(cleaned_df.describe())
    cleaned_df['age'] = pd.to_numeric(cleaned_df['age'])
    cleaned_df['age'] = cleaned_df['age'].fillna(cleaned_df['age'].median())

    cleaned_df['rating'] = pd.to_numeric(cleaned_df['rating'])
    cleaned_df['rating'] = cleaned_df['rating'].fillna(cleaned_df['rating'].median())
   
    col_mapper = {
    'bust size': 'bust_size',
    'weight': 'weight_kg',
    'rating': 'review_rating',
    'rented for': 'rented_for',
    'body type': 'body_type',
    'height': 'height_meters',
    'size': 'product_size',
    'category': 'product_category_old',
    'age': 'age',
    }
    cleaned_df.rename(col_mapper, axis=1, inplace=True)

     #group the products into tops, bottoms and other
    mapper = {
    0: 'product_category', 
    }

    temp_df = cleaned_df['product_category_old'].apply(parse_product_category).rename(mapper, axis=1)
    temp_df = temp_df[~temp_df.product_category.str.contains("other")] #contains sizes from 35 to 58 

    #temp_df.info()
    cleaned_df.info()
    cleaned_df = cleaned_df.join(temp_df)
    cleaned_df.drop(['product_category_old'], axis=1, inplace=True)

    mapper = {
    0: 'bust_size_num_us', 
    1: 'bust_size_cat'
    }
    temp_df = cleaned_df['bust_size'].apply(parse_bust_size).rename(mapper, axis=1)
    temp_df['bust_size_num_us'] = pd.to_numeric(temp_df['bust_size_num_us'])
    cleaned_df = cleaned_df.join(temp_df)
    cleaned_df.drop(['bust_size'], axis=1, inplace=True)
    cleaned_df.head()


    mapper = {
    0: 'bust_size_num_eu', 
    }

    #Convert US into EU bustsize
    #cleaned_df.product_size_old = cleaned_df['bust_size_num'].astype(object)
    temp_df = cleaned_df['bust_size_num_us'].apply(convert_bust_size).rename(mapper, axis=1)
    temp_df['bust_size_num_eu'] = pd.to_numeric(temp_df['bust_size_num_eu'])
    cleaned_df = cleaned_df.join(temp_df)
    cleaned_df.drop(['bust_size_num_us'], axis=1, inplace=True)

    cleaned_df['bmi'] = cleaned_df['weight_kg'] / np.power(cleaned_df['height_meters'], 2)
    
    # mapper = {
    # 0: 'product_size_new', 
    # }

    # cleaned_df.product_size = cleaned_df['product_size'].astype(object)
    # cleaned_df.info()
    # temp_df = cleaned_df['product_size'].apply(parse_product_size).rename(mapper, axis=1)

    # cleaned_df = cleaned_df.join(temp_df)
    #cleaned_df.drop(['product_size_old'], axis=1, inplace=True)

    #cleaned_df = cleaned_df.dropna()
   # cleaned_df.bmi = cleaned_df.bmi.fillna(0).astype('float64')'
    cleaned_df= cleaned_df[~cleaned_df.isin([np.nan, np.inf, -np.inf]).any(1)]
    #cleaned_df = cleaned_df.drop(cleaned_df[(cleaned_df.bmi == 'inf') & (cleaned_df.bmi == 0)].index)
    print(cleaned_df.describe())
    cleaned_df.to_csv(fileoutput, index=False)

