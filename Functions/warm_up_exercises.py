"""
@file warm_up_exercise.py
@brief Returns a 5x5 identity matrix and provides helper functions for linear regression.
"""

import numpy as np

def warm_up_exercise1():
    """
    @brief Create and return a 5x5 identity matrix.

    @return np.ndarray Identity matrix (5x5)
    """
    return np.eye(5)  # np.eye cria uma matriz identidade de tamanho 5x5

def warm_up_exercise2(m=5):
    """
    @brief Cria um vetor coluna de 1s, utilizado como termo de bias (intercepto) em regressão linear.

    @param m: int
        Número de exemplos (linhas).

    @return np.ndarray
        Vetor de shape (m, 1) com todos os valores iguais a 1.
    """
    return np.ones((m, 1))  # Cria um vetor coluna (m linhas, 1 coluna) com todos os valores iguais a 1

def warm_up_exercise3(x):
    """
    @brief Adiciona uma coluna de 1s (bias) ao vetor de entrada x.

    @param x: np.ndarray
        Vetor unidimensional de shape (m,)

    @return np.ndarray
        Matriz de shape (m, 2), com a primeira coluna sendo 1s (bias) e a segunda os valores de x.
    """
    m = x.shape[0]  # Obtém o número de exemplos (linhas)
    x = np.reshape(x, (m, 1))  # Garante que x seja vetor coluna
    bias = np.ones((m, 1))  # Cria a coluna de 1s (bias)
    return np.hstack((bias, x))  # Concatena bias e x horizontalmente (lado a lado)

def warm_up_exercise4(X, theta):
    """
    @brief Realiza a multiplicação matricial entre X e θ, simulando h(θ) = X @ θ.

    @param X: np.ndarray
        Matriz de entrada de shape (m, n)

    @param theta: np.ndarray
        Vetor de parâmetros de shape (n,)

    @return np.ndarray
        Vetor de predições (m,)
    """
    return X @ theta  # Multiplicação matricial (produto escalar entre X e os pesos theta)

def warm_up_exercise5(predictions, y):
    """
    @brief Calcula o vetor de erros quadráticos (squared errors) entre as predições e os valores reais.

    @param predictions: np.ndarray
        Vetor de predições (m,)

    @param y: np.ndarray
        Vetor de valores reais (m,)

    @return np.ndarray
        Vetor com os erros quadráticos: (pred - y)^2
    """
    return (predictions - y) ** 2  # Erros quadráticos: diferença ao quadrado entre predição e valor real

def warm_up_exercise6(errors):
    """
    @brief Calcula o custo médio (mean cost) a partir dos erros quadráticos.

    @param errors: np.ndarray
        Vetor de erros quadráticos (m,)

    @return float
        Custo médio (mean cost)
    """
    return np.mean(errors) / 2  # Custo médio = média dos erros quadráticos dividido por 2

def warm_up_exercise7(X, y, theta):
    """
    @brief Calcula o custo médio (mean cost) para um modelo de regressão linear.

    @param X: np.ndarray
        Matriz de entrada de shape (m, n)

    @param y: np.ndarray
        Vetor de valores reais (m,)

    @param theta: np.ndarray
        Vetor de parâmetros de shape (n,)

    @return float
        Custo médio (mean cost)
    """
    predictions = warm_up_exercise4(X, theta)         # Passo 1: h(θ) = X @ θ
    errors = warm_up_exercise5(predictions, y)        # Passo 2: Erros quadráticos
    return warm_up_exercise6(errors)                  # Passo 3: Custo médio (mean cost)
