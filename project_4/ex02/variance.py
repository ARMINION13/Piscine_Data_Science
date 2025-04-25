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
    
    #Normalize
    for col in df_train.columns:
        df_train[col] = (df_train[col] / df_train[col].abs().max())
    
    var = df_train.var().sort_values(ascending=False)

    #varianza antes de la suma acumulativa
    print("Variances (Percentage):")
    print(var / var.sum())

    rng = range(0,31)
    var = (var.cumsum() / var.sum()) * 100

    print("Cumulative Variances (Porcentage):")
    print(var)

    fig, var_chart = plt.subplots(figsize=(14, 12))
    var_chart.grid()
    var_chart.plot(var)
    var_chart.set_xticklabels(range(0,31))


data_train = get_dataframe('../Train_knight.csv')

correlation_heatmap( data_train )

plt.tight_layout()
plt.show()