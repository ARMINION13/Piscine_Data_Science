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
    df_train = df_train.drop(columns=['Unnamed: 0'])

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(df_train[df_train.columns[:-1]], df_train['knight'])

    tree.plot_tree(dtree, feature_names=df_train.columns)

    return(dtree)
    
def predict_tree(model, df_test):

    df_test = df_test.drop(columns=['Unnamed: 0'])
    predictions_binary = model.predict(df_test[df_test.columns[:-1]])

    return predictions_binary

def get_matrix( truth : list, predict : list):

    matrix = []
    matrix.append([0, 0])
    matrix.append([0, 0])

    for i in range(0, len(truth)):
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

data_train = get_dataframe(sys.argv[1])
data_val = get_dataframe(sys.argv[2])
data_val['knight'] = data_val['knight'].map({'Jedi': 0, 'Sith': 1})

tree_model = train_tree( data_train )

truth = data_val['knight'].tolist()
predict = predict_tree( tree_model, data_val )

matrix = get_matrix( truth, predict )
print_matrix ( matrix )
confusion_matrix_chart( matrix )

plt.tight_layout()
plt.show()