import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)
    return df

def correlation_factor( df_train ):

    df_train['knight'] = df_train['knight'].map({'Jedi': 1, 'Sith': 0})
    correlation = df_train.corr(numeric_only=True)['knight']
    print(correlation.to_frame())

data_train = get_dataframe('../Train_knight.csv')

correlation_factor( data_train )