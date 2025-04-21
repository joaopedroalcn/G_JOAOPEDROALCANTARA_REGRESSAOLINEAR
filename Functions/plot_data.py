"""
@file plot_data.py
@brief Plots the data points.
"""

import matplotlib.pyplot as plt
import numpy as np  

def plot_data(x, y):
    """
    @brief Plot training data as red crosses.

    @param x np.ndarray Independent variable (population)
    @param y np.ndarray Dependent variable (profit)
    """
    plt.figure()
    plt.plot(x, y, 'rx', markersize=5)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Training Data')
    plt.grid(True)
    plt.show()

def load_data():
    """
    Carrega os dados do arquivo 'ex1data1.txt' e retorna X (entrada) e y (saída).
    """
    data_path = 'Data/ex1data1.txt'  # Caminho para o arquivo de dados
    data = np.loadtxt(data_path, delimiter=',')
    X = data[:, 0]  # Primeira coluna é X (população)
    y = data[:, 1]  # Segunda coluna é y (lucro)
    return X, y