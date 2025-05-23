import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def get_dataframe( dir : str ):

    df = pd.read_csv( dir )

    return df

def train_KNN( df_train ):

    knn = KNeighborsClassifier( n_neighbors=2 )
    knn.fit(df_train[df_train.columns[:-1]], df_train['knight'])

    return knn

def predict_KNN(model, df_test):

    predictions_binary = model.predict(df_test[df_test.columns])

    return predictions_binary

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

def iterative_accuracy( df_train, df_test ):

    #Inicializo las variables
    truth = get_1D_data("../Truth_Knight.txt")
    accuracy = []
    k_values = []

    #Saco la accuracy por n_neighbors
    for k in range(1, 30):
        
        k_values.append(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(df_train[df_train.columns[:-1]], df_train['knight'])

        predictions_binary = knn.predict(df_test[df_test.columns])
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

#Normalizo los dataframes
scaler = StandardScaler()
data_train[data_train.columns[:-1]] = scaler.fit_transform(data_train[data_train.columns[:-1]])
data_test[data_train.columns[:-1]] = scaler.transform(data_test[data_test.columns])

KNN_model = train_KNN( data_train )

predict = predict_KNN( KNN_model, data_test )

with open('../KNN.txt', 'w') as f:
    for item in predict:
        if item == 0:
            f.write(f"Jedi\n")
        if item == 1:
            f.write(f"Sith\n")

accuracy, k_values = iterative_accuracy( data_train, data_test )
k_accuracy_chart(accuracy, k_values)

plt.tight_layout()
plt.show()