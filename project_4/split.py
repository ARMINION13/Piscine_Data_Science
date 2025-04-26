import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)
    return df

data_train = get_dataframe('../Train_knight.csv')
data_test = get_dataframe('../Test_knight.csv')

split_percentage = len(data_test.index) / len(data_train.index)

data_valid = data_train.sample(frac = split_percentage)
new_data_train = data_train.drop(data_valid.index)

data_valid.to_csv('../Validation_knight.csv')
new_data_train.to_csv('../Training_knight.csv')

