import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # editing visualizations

def create_df():
    df = pd.read_json('./Data/modcloth_final_data.json', lines=True)
    df.columns = ['item_id', 'waist', 'size', 'quality', 'cup_size', 'hips','bra_size','category', 'bust', 'height', 'user_name', 'length','fit', 'user_id', 'shoe_size', 'shoe_width', 'review_summary', 'review_text']


    #check missing data:
    missing_data = pd.DataFrame({'total_missing': df.isnull().sum(), 'perc_missing': (df.isnull().sum()/82790)*100})
    #print(missing_data)

    #change types. 
    df.bra_size = df.bra_size.fillna('Unknown')
    df.bra_size = df.bra_size.astype(str)
    df.at[37313,'bust'] = '38'
    df.bust = df.bust.fillna(0).astype(int)
    df.category = df.category.astype(str)
    df.cup_size.fillna('Unknown', inplace=True)
    df.cup_size = df.cup_size.astype(str)
    df.fit = df.fit.astype(str)

    def get_cms(x):
        if type(x) == type(1.0):
            return
        try: 
            return (int(x[0])*30.48) + (int(x[4:-2])*2.54)
        except:
            return (int(x[0])*30.48)

    df.height = df.height.apply(get_cms)
    #print(df[df.height.isnull()].head(20))


    lingerie_cond = (((df.bra_size != 'Unknown') | (df.cup_size != 'Unknown')) & (df.height.isnull()) & (df.hips.isnull()) &
        (df.shoe_size.isnull()) & (df.shoe_width.isnull()) & (df.waist.isnull()))
    shoe_cond = ((df.bra_size == 'Unknown') & (df.cup_size == 'Unknown') & (df.height.isnull()) & (df.hips.isnull()) &
        ((df.shoe_size.notnull()) | (df.shoe_width.notnull())) & (df.waist.isnull()))
    dress_cond = ((df.bra_size == 'Unknown') & (df.cup_size == 'Unknown') & (df.height.isnull()) & ((df.hips.notnull()) | (df.waist.notnull())) &
        (df.shoe_size.isnull()) & (df.shoe_width.isnull()))
    #print(len(df[lingerie_cond]))   # To check if these items add up in the final column we are adding.
    #print(len(df[shoe_cond]))
    #print(len(df[dress_cond]))
    df['first_time_user'] = (lingerie_cond | shoe_cond | dress_cond)
    #print("Column added!")
    #print("Total transactions by first time users who bought bra, shoes, or a dress: " + str(sum(df.first_time_user)))
    #print("Total first time users: " + str(len(df[(lingerie_cond | shoe_cond | dress_cond)].user_id.unique())))

    # Handling hips column
    df.hips = df.hips.fillna(-1.0)
    bins = [-5,0,31,37,40,44,75]
    labels = ['Unknown','XS','S','M', 'L','XL']
    df.hips = pd.cut(df.hips, bins, labels=labels)

    # Handling length column
    missing_rows = df[df.length.isnull()].index
    df.drop(missing_rows, axis = 0, inplace=True)

    # Handling quality
    missing_rows = df[df.quality.isnull()].index
    df.drop(missing_rows, axis = 0, inplace=True)
    df.quality = df.quality.astype('string')

    from pandas.api.types import CategoricalDtype
    shoe_widths_type = CategoricalDtype(categories=['Unknown','narrow','average','wide'], ordered=True)

    df.review_summary = df.review_summary.fillna('Unknown')
    df.review_text = df.review_text.fillna('Unkown')
    df.shoe_size = df.shoe_size.fillna('Unknown')
    df.shoe_size = df.shoe_size.astype(str)
    df.shoe_width = df.shoe_width.fillna('Unknown')
    df.shoe_width = df.shoe_width.astype(shoe_widths_type)
    df.drop(['waist', 'bust', 'user_name'], axis=1, inplace=True)
    missing_rows = df[df.height.isnull()].index
    df.drop(missing_rows, axis = 0, inplace=True)

    return df

