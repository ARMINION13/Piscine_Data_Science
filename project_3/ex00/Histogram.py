import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)
    return df

def make_histogram( df_test ):

    fig, histplot = plt.subplots(6, 5, figsize=(20, 20))

    for y in range(0, 6):
        for x in range(0, 5):
            col = df_test.columns[(5 * y) + x]
            histplot[y, x].hist(x=df_test[col], bins=40, color='lightgreen', edgecolor=None)
            histplot[y, x].set_title(df_test.columns[(5 * y) + x])

def make_compare_histogram( df_train ):

    fig, histplot = plt.subplots(6, 5, figsize=(20, 20))
    Jedi = df_train[df_train['knight'] == 'Jedi']
    Sith = df_train[df_train['knight'] == 'Sith']

    for y in range(0, 6):
        for x in range(0, 5):
            col = df_train.columns[(5 * y) + x]
            histplot[y, x].hist(x=Jedi[col], bins=40, color='lightblue', edgecolor=None, alpha=0.5)
            histplot[y, x].hist(x=Sith[col], bins=40, color='lightcoral', edgecolor=None, alpha=0.5)
            histplot[y, x].set_title(df_train.columns[(5 * y) + x])

data_test = get_dataframe('../Test_knight.csv')
data_train = get_dataframe('../Train_knight.csv')

make_histogram( data_test )
make_compare_histogram( data_train )

plt.tight_layout()
plt.show()