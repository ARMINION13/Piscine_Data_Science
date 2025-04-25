import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)

    return df

def print_multicollinearity( df_train ):

    df_train['knight'] = df_train['knight'].map({'Jedi': 0, 'Sith': 1})

    cols = df_train[df_train.columns]

    VIF_data = pd.DataFrame()
    VIF_data["SKILLS"] = cols.columns

    VIF_data["VIF"] = [vif(cols.values, i) for i in range(len(cols.columns))]
    VIF_data["TOLERANCE"] = [(1 / vif(cols.values, i)) for i in range(len(cols.columns))]

    print("BEFORE DROPING HIGH VIF")
    print("----------------------------------------------------")
    print("                                                    ")
    print(VIF_data)

    VIF_data = VIF_data[VIF_data['TOLERANCE'] > 0.2]

    print("                                                    ")     
    print("AFTER DROPING HIGH VIF")
    print("----------------------------------------------------")
    print("                                                    ")     
    print(VIF_data)

data_train = get_dataframe('../Train_knight.csv')

print_multicollinearity( data_train )
