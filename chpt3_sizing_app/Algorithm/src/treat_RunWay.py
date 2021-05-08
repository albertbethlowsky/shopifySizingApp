
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import json

def load_data(fp):
    with open(fp) as fid:
        series = (pd.Series(json.loads(s)) for s in fid)
        return pd.concat(series, axis=1).T

def feet_to_meters(s):
    feet, inches = map(int, s.replace("'", '').replace("\"", '').split())
    return feet * 0.3048 + inches * 0.0254

def pounds_to_kilos(s):
    return int(s.replace('lbs', '')) * 0.45359237

def create_csv():
    df = load_data('./renttherunway_final_data.json')

    print(df.head(20))

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


create_csv()