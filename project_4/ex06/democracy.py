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

    predictions_KNN = model.predict(df_test[df_test.columns[:-1]])

    return predictions_KNN

#______________________________Decision_Tree________________________________#
def train_tree( df_train ):

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(df_train[df_train.columns[:-1]], df_train['knight'])

    tree.plot_tree(dtree, feature_names=df_train.columns)

    return dtree
    
def predict_tree( model, df_test ):

    predictions_binary = model.predict(df_test[df_test.columns[:-1]])

    return predictions_binary


#____________________________Random_Forest__________________________________#
def train_random_forest( df_train ):

    random_forest = RandomForestClassifier( n_estimators=100 )
    random_forest.fit( df_train[df_train.columns[:-1]], df_train['knight'] )    

    return random_forest

def predict_random_forest( model, df_test ):

    prediction_forest = model.predict( df_test[df_test.columns[:-1]] )

    return prediction_forest

#___________________________________Utils___________________________________#
def get_1D_data( dir : str):

    binary_data = []

    with open(dir, 'r') as f:
        data = f.read().split('\n')
    
    for d in data:
        if d == 'Jedi':
            binary_data.append(0)
        if d == 'Sith':
            binary_data.append(1)
    
    return binary_data

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)

    return df

def get_matrix( truth : list, predict : list):

    matrix = []
    matrix.append([0, 0])
    matrix.append([0, 0])

    for i in range(0, len(predict)):
        if truth[i] == 0 and predict[i] == 0: #Verdadero positivo
            matrix[0][0] = matrix[0][0] + 1
        if truth[i] == 0 and predict[i] == 1: #Falso positivos
            matrix[0][1] = matrix[0][1] + 1
        if truth[i] == 1 and predict[i] == 0: #Falso negativos
            matrix[1][0] = matrix[1][0] + 1
        if truth[i] == 1 and predict[i] == 1: #Verdadero negativo
            matrix[1][1] = matrix[1][1] + 1

    return matrix

def print_matrix( matrix : list ):

    #precision
    p_jedi = round(matrix[0][0] / (matrix[0][0] + matrix[1][0]), 2)
    p_sith = round(matrix[1][1] / (matrix[1][1] + matrix[0][1]), 2)

    #recall
    r_jedi = round(matrix[0][0] / (matrix[0][0] + matrix[0][1]), 2)
    r_sith = round(matrix[1][1] / (matrix[1][1] + matrix[1][0]), 2)

    #F1-Score
    f1_jedi = round(2 * (p_jedi * r_jedi) / (p_jedi + r_jedi), 2)
    f1_sith = round(2 * (p_sith * r_sith) / (p_sith + r_sith), 2)

    #total
    t_jedi = matrix[0][0] + matrix[1][0]
    t_sith = matrix[1][1] + matrix[0][1]

    #accuracy

    f1_acc = round((f1_jedi + f1_sith) / 2, 2)
    total_acc =  t_jedi + t_sith

    print("        Precision    Recall    F1-score    Total")
    print("                                                ")
    print(f"Jedi   {p_jedi:>10}{r_jedi:>10}{f1_jedi:>12}{t_jedi:>9}")
    print(f"Sith   {p_sith:>10}{r_sith:>10}{f1_sith:>12}{t_sith:>9}")
    print("                                                ")
    print(f"accuracy                       {f1_acc:>8}{total_acc:>9}")
    print("                                                ")
    print(matrix[0])
    print(matrix[1])

def confusion_matrix_chart( matrix ):

    fig, matrix_chart = plt.subplots(figsize=(7, 4))

    sns.heatmap(matrix, annot=True, fmt="d", cmap="viridis"
    , xticklabels=[0, 1], yticklabels=[0, 1])


#_____________________________DEMOCRACY_____________________________________#

def Democracy( p_1, p_2, p_3 ):

    final_predictions = []

    for p in range(0, (len(p_1) - 1)):

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

    with open('../Vortix.txt', 'w') as f:
        for item in final_predictions:
            f.write(f"{item}\n")


#_________________________MAIN___________________________________#

data_train = get_dataframe(sys.argv[1])
data_test = get_dataframe(sys.argv[2])
data_train['knight'] = data_train['knight'].map({'Jedi': 0, 'Sith': 1})
data_test['knight'] = data_test['knight'].map({'Jedi': 0, 'Sith': 1})
data_train = data_train.drop(columns=['Unnamed: 0'])
data_test = data_test.drop(columns=['Unnamed: 0'])

# #Normalizo los dataframes
# scaler = StandardScaler()
# data_train[data_train.columns[:-1]] = scaler.fit_transform(data_train[data_train.columns[:-1]])
# data_test[data_train.columns[:-1]] = scaler.transform(data_test[data_test.columns[:-1]])

tree_model = train_tree( data_train )
KNN_model = train_KNN( data_train )
RF_model = train_random_forest( data_train )

truth = data_test['knight'].tolist()
p_KNN = predict_KNN( KNN_model, data_test )
p_tree = predict_tree( tree_model, data_test )
p_RanFor = predict_random_forest( RF_model, data_test)


Democracy(p_KNN, p_tree, p_RanFor)
predict = get_1D_data('../Vortix.txt')

matrix = get_matrix( truth, predict )
print_matrix ( matrix )
confusion_matrix_chart( matrix )

plt.tight_layout()
plt.show()