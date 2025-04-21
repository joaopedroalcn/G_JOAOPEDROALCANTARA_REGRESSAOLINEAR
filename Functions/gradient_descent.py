"""
@file gradient_descent.py
@brief Implementa o algoritmo de descida do gradiente para regressão linear.
"""

import numpy as np
from compute_cost import compute_cost


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Executa a descida do gradiente para minimizar a função de custo J(θ)
    no contexto de regressão linear.
    """
    # Obtem o número de amostras
    m = len(y)
    
    # Inicializa o vetor de custo J_history
    J_history = np.zeros(num_iters)
    
    # Inicializa o vetor theta_history
    theta_history = np.zeros((num_iters + 1, theta.shape[0]))
    
    # Armazena os parâmetros iniciais
    theta_history[0] = theta

    for i in range(num_iters):
        # Calcula as previsões (hipótese)
        predictions = X.dot(theta)

        # Calcula o erro
        erro = predictions - y

        # Calcula o gradiente
        gradient = (1 / m) * X.T.dot(erro)

        # Atualiza os parâmetros theta
        theta = theta - alpha * gradient

        # Armazena o custo da iteração atual
        J_history[i] = compute_cost(X, y, theta)

        # Armazena os parâmetros atualizados
        theta_history[i + 1] = theta

    return theta, J_history, theta_history
