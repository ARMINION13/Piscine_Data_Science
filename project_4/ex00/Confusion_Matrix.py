import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

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
    t_jedi = matrix[0][0] + matrix[0][1]
    t_sith = matrix[1][1] + matrix[1][0]

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

truth = get_1D_data(sys.argv[2])
predict = get_1D_data(sys.argv[1])

matrix = get_matrix( truth, predict )
print_matrix ( matrix )
confusion_matrix_chart( matrix )

plt.tight_layout()
plt.show()