
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

def parse_product_category(s):
    tops = ['blazer', 'blouse', 'blouson', 'bomber', 'buttondown','cami', 'cardigan', 'coat', 'crewneck', 'henley', 'hoodie', 'jacket', 'overcoat', 'parka', 'peacoat', 'pullover', 'shirt', 'sweater', 'sweater', 'sweatershirt', 'sweatshirt', 'tank', 'tee', 'tops', 'trench', 't-shirt', 'turtleneck', 'vest']
    bottoms = ['culotte', 'culottes', 'jeans', 'jogger', 'legging', 'leggings', 'overalls', 'pant', 'pants', 'skirt', 'skirts', 'skort', 'sweatpants', 'trouser', 'trousers']
    other = ['caftan', 'cape', 'combo', 'dress', 'duster', 'for', 'frock', 'gown', 'jumpsuit', 'kaftan', 'kimono', 'knit', 'maxi', 'midi','mini', 'poncho', 'print', 'romper', 'sheath', 'shift', 'shirtdress', 'suit', 'tight', 'tunic']
    if any(s in index for index in tops):
        return pd.Series(data=['top'])
    elif any(s in index for index in bottoms):
        return pd.Series(data=['bottom'])
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
    
    cleaned_df['weight'] = (cleaned_df['weight']
                            .fillna('0lbs')
                            .apply(pounds_to_kilos))

    cleaned_df['user_id'] = pd.to_numeric(cleaned_df['user_id'])
    cleaned_df['bust size'] = cleaned_df['bust size'].fillna(cleaned_df['bust size'].value_counts().index[0])
    cleaned_df['body type'] = cleaned_df['body type'].fillna(cleaned_df['body type'].value_counts().index[0])
    cleaned_df['item_id'] = pd.to_numeric(cleaned_df['item_id'])
    cleaned_df['size'] = pd.to_numeric(cleaned_df['size'])

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
    'category': 'product_category_old',
    'height': 'height_meters',
    'size': 'product_size',
    'age': 'age',
    }
    cleaned_df.rename(col_mapper, axis=1, inplace=True)


    mapper = {
    0: 'bust_size_num', 
    1: 'bust_size_cat'
    }
    temp_df = cleaned_df['bust_size'].apply(parse_bust_size).rename(mapper, axis=1)
    temp_df['bust_size_num'] = pd.to_numeric(temp_df['bust_size_num'])
    
    cleaned_df = cleaned_df.join(temp_df)

    cleaned_df.drop(['bust_size'], axis=1, inplace=True)
    cleaned_df['bmi'] = cleaned_df['weight_kg'] / np.power(cleaned_df['height_meters'], 2)
    cleaned_df.drop(['weight_kg', 'height_meters'], axis=1, inplace=True)

    mapper = {
    0: 'product_category', 
    }
    temp_df = cleaned_df['product_category_old'].apply(parse_product_category).rename(mapper, axis=1)
    cleaned_df = cleaned_df.join(temp_df)
    cleaned_df.drop(['product_category_old'], axis=1, inplace=True)

    #cleaned_df = cleaned_df.dropna()
   # cleaned_df.bmi = cleaned_df.bmi.fillna(0).astype('float64')'
    cleaned_df= cleaned_df[~cleaned_df.isin([np.nan, np.inf, -np.inf]).any(1)]
    #cleaned_df = cleaned_df.drop(cleaned_df[(cleaned_df.bmi == 'inf') & (cleaned_df.bmi == 0)].index)
    cleaned_df.to_csv(fileoutput, index=False)

create_csv('C:/Users/Frederik/Desktop/shopifySizingApp/chpt3_sizing_app/Algorithm/Data/renttherunway_final_data.json', 'C:/Users/Frederik/Desktop/shopifySizingApp/chpt3_sizing_app/Algorithm/Data/clean_runway.csv')