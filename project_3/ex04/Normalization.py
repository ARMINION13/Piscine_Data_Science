import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)
    return df

def make_scatters( df_train ):

    fig, scatter = plt.subplots(figsize=(6, 5))

    # Jedis vs Siths

        #Friendship vs Repulse

    Jedi = df_train[df_train['knight'] == 'Jedi']
    Sith = df_train[df_train['knight'] == 'Sith']

    scatter.scatter(Jedi['Sensitivity'], Jedi['Strength'], color=None, edgecolor='blue', alpha=0.7)
    scatter.scatter(Sith['Sensitivity'], Sith['Strength'], color=None, edgecolor='red', alpha=0.7)
    scatter.set_xlabel('Friendship')
    scatter.set_ylabel('Strength')

def ft_normalize( df_train ):

    print(df_train)

    for col in df_train.columns :
        if col != 'knight':
            df_train[col] = (df_train[col] / df_train[col].abs().max())

    print(df_train)

data_train = get_dataframe('../Train_knight.csv')

ft_normalize( data_train )
make_scatters( data_train )

plt.tight_layout()
plt.show()