import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)

    return df

def train_tree( df_train ):

    df_train['knight'] = df_train['knight'].map({'Jedi': 0, 'Sith': 1})

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(df_train[df_train.columns[:-1]], df_train['knight'])

    tree.plot_tree(dtree, feature_names=df_train.columns)

    return(dtree)
    
def predict_tree(model, df_test):

    predictions_binary = model.predict(df_test[df_test.columns])

    return predictions_binary

data_train = get_dataframe(sys.argv[1])
data_val = get_dataframe(sys.argv[2])

tree_model = train_tree( data_train )

predict = predict_tree( tree_model, data_val )

with open('../Tree.txt', 'w') as f:
    for item in predict:
        if item == 0:
            f.write(f"Jedi\n")
        if item == 1:
            f.write(f"Sith\n")

plt.tight_layout()
plt.show()