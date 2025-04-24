import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)
    return df

def make_scatters( df_test, df_train ):

    fig, scatter = plt.subplots(2, 2, figsize=(12, 10))

    # Jedis vs Siths

        #Sensitivity vs Strength

    Jedi = df_train[df_train['knight'] == 'Jedi']
    Sith = df_train[df_train['knight'] == 'Sith']

    scatter[0, 0].scatter(Jedi['Sensitivity'], Jedi['Strength'], color=None, edgecolor='blue', alpha=0.7)
    scatter[0, 0].scatter(Sith['Sensitivity'], Sith['Strength'], color=None, edgecolor='red', alpha=0.7)
    scatter[0, 0].set_xlabel('Friendship')
    scatter[0, 0].set_ylabel('Strength')

        #Friendship vs Repulse

    scatter[0, 1].scatter(Jedi['Friendship'], Jedi['Repulse'], color=None, edgecolor='blue', alpha=0.7)
    scatter[0, 1].scatter(Sith['Friendship'], Sith['Repulse'], color=None, edgecolor='red', alpha=0.7)
    scatter[0, 1].set_xlabel('Friendship')
    scatter[0, 1].set_ylabel('Repulse')

    # Knights

        #Sensitivity vs Strength

    scatter[1, 0].scatter(df_test['Sensitivity'], df_test['Strength'], color=None, edgecolor='green', alpha=0.5)
    scatter[1, 0].set_xlabel('Sensitivity')
    scatter[1, 0].set_ylabel('Strength')

        #Friendship vs Repulse

    scatter[1, 1].scatter(df_test['Friendship'], df_test['Repulse'], color=None, edgecolor='green', alpha=0.5)
    scatter[1, 1].set_xlabel('Friendship')
    scatter[1, 1].set_ylabel('Repulse')


data_test = get_dataframe('../Test_knight.csv')
data_train = get_dataframe('../Train_knight.csv')

make_scatters( data_test, data_train )

plt.tight_layout()
plt.show()