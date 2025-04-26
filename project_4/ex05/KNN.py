import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from sklearn.neighbors import KNeighborsClassifier

def get_dataframe( dir : str ):

    df = pd.read_csv(dir)

    return df

def train_KNN( df_train ):

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(df_train[df_train.columns[:-1]], df_train['knight'])

    return knn

def predict_KNN(model, df_test):

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


def iterative_accuracy( df_train, df_test ):

    #Normalizo los dataframes
    # for col in df_train.columns[:-1]:
    #     df_train[col] = (df_train[col] / df_train[col].abs().max())
    #     df_test[col] = (df_test[col] / df_test[col].abs().max())

    #Inicializo las variables
    truth = data_test['knight'].tolist()
    accuracy = []
    k_values = []

    #Saco la accuracy por n_neighbors
    for k in range(1, 30):
        
        k_values.append(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(df_train[df_train.columns[:-1]], df_train['knight'])

        predictions_binary = knn.predict(df_test[df_test.columns[:-1]])
        matrix = get_matrix( truth, predictions_binary )
         #precision
        p_jedi = round(matrix[0][0] / (matrix[0][0] + matrix[1][0]), 2)
        p_sith = round(matrix[1][1] / (matrix[1][1] + matrix[0][1]), 2)

        #recall
        r_jedi = round(matrix[0][0] / (matrix[0][0] + matrix[0][1]), 2)
        r_sith = round(matrix[1][1] / (matrix[1][1] + matrix[1][0]), 2)

        #F1-Score
        f1_jedi = round(2 * (p_jedi * r_jedi) / (p_jedi + r_jedi), 2)
        f1_sith = round(2 * (p_sith * r_sith) / (p_sith + r_sith), 2)
        accuracy.append(round((f1_jedi + f1_sith) / 2, 2))
    
    print(accuracy)
    print(k_values)
    return accuracy, k_values

def k_accuracy_chart( accuracy, k_values):

    fig, k_accuracy_chart = plt.subplots(figsize=(14, 12))
    k_accuracy_chart.plot(k_values, accuracy, linewidth=3)
    k_accuracy_chart.set_ylabel('accuracy')
    k_accuracy_chart.set_xlabel('k values')


#_________________________MAIN___________________________________#

data_train = get_dataframe(sys.argv[1])
data_test = get_dataframe(sys.argv[2])
data_train['knight'] = data_train['knight'].map({'Jedi': 0, 'Sith': 1})
data_test['knight'] = data_test['knight'].map({'Jedi': 0, 'Sith': 1})
#data_train = data_train.drop(columns=['Unnamed: 0'])
data_test = data_test.drop(columns=['Unnamed: 0'])





# KNN_model = train_KNN( data_train )

# truth = data_test['knight'].tolist()
# predict = predict_KNN( KNN_model, data_test )

# matrix = get_matrix( truth, predict )
# print_matrix ( matrix )
# confusion_matrix_chart( matrix )

accuracy, k_values = iterative_accuracy(data_train, data_test)
k_accuracy_chart(accuracy, k_values)

plt.tight_layout()
plt.show()