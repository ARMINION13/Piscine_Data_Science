import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)

    return df

def correlation_heatmap( df_train ):

    df_train['knight'] = df_train['knight'].map({'Jedi': 0, 'Sith': 1})
    correlation = df_train.corr()

    fig, heatmap_chart = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(correlation, cmap="Reds_r" ,annot=None
    , xticklabels=df_train.columns, yticklabels=df_train.columns)


data_train = get_dataframe('../Train_knight.csv')

correlation_heatmap( data_train )

plt.tight_layout()
plt.show()