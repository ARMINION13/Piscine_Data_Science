import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


#___________________________________KNN_____________________________________#
def train_KNN( df_train ):

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(df_train[df_train.columns[:-1]], df_train['knight'])

    return knn

def predict_KNN( model, df_test ):

    predictions_KNN = model.predict(df_test[df_test.columns])

    return predictions_KNN

#______________________________Decision_Tree________________________________#
def train_tree( df_train ):

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(df_train[df_train.columns[:-1]], df_train['knight'])

    return dtree
    
def predict_tree( model, df_test ):

    predictions_binary = model.predict(df_test[df_test.columns])

    return predictions_binary


#____________________________Random_Forest__________________________________#
def train_random_forest( df_train ):

    random_forest = RandomForestClassifier( n_estimators=100 )
    random_forest.fit( df_train[df_train.columns[:-1]], df_train['knight'] )    

    return random_forest

def predict_random_forest( model, df_test ):

    prediction_forest = model.predict( df_test[df_test.columns] )

    return prediction_forest

#___________________________________Utils___________________________________#

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)

    return df


#_____________________________DEMOCRACY_____________________________________#

def Democracy( p_1, p_2, p_3 ):

    final_predictions = []

    for p in range(0, (len(p_1))):

        vote = 0

        if p_1[p] == 0:
            vote = vote + 1
        if p_2[p] == 0:
            vote = vote + 1        
        if p_3[p] == 0:
            vote = vote + 1

        if vote > 1:
            final_predictions.append('Jedi')
        elif vote <= 1:
            final_predictions.append('Sith')

    with open('../Voting.txt', 'w') as f:
        for item in final_predictions:
            f.write(f"{item}\n")


#_________________________MAIN___________________________________#

data_train = get_dataframe(sys.argv[1])
data_test = get_dataframe(sys.argv[2])
data_train['knight'] = data_train['knight'].map({'Jedi': 0, 'Sith': 1})

# #Normalizo los dataframes
# scaler = StandardScaler()
# data_train[data_train.columns[:-1]] = scaler.fit_transform(data_train[data_train.columns[:-1]])
# data_test[data_train.columns[:-1]] = scaler.transform(data_test[data_test.columns[:-1]])

tree_model = train_tree( data_train )
KNN_model = train_KNN( data_train )
RF_model = train_random_forest( data_train )

p_KNN = predict_KNN( KNN_model, data_test )
p_tree = predict_tree( tree_model, data_test )
p_RanFor = predict_random_forest( RF_model, data_test)


Democracy(p_KNN, p_tree, p_RanFor)