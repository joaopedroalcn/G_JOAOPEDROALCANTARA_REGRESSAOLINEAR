import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Adiciona o diretório Functions ao caminho de busca do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compute_cost import compute_cost

def plot_contour(X, y, theta_0_vals, theta_1_vals, J_vals):
    """
    Gera o gráfico de contorno da função de custo J(θ) em relação aos parâmetros θ0 e θ1.

    @param X np.ndarray: Matriz de características (incluindo o termo de bias).
    @param y np.ndarray: Vetor de valores de saída.
    @param theta_0_vals np.ndarray: Valores para o parâmetro θ0.
    @param theta_1_vals np.ndarray: Valores para o parâmetro θ1.
    @param J_vals np.ndarray: Valores da função de custo J(θ) para cada par de (θ0, θ1).
    """
    # Plotando o gráfico de contorno
    plt.contour(theta_0_vals, theta_1_vals, J_vals, levels=np.logspace(0, 5, 20), cmap='jet')
    plt.xlabel('θ0')
    plt.ylabel('θ1')
    plt.title('Gráfico de Contorno da Função de Custo J(θ)')
    plt.colorbar()  # Adiciona uma barra de cores
    plt.grid(True)
    
    plt.savefig("Figures/grafico_contorno.png")  # Salva a imagem como .png

    plt.show()

def generate_contour_data(X, y, theta_0_vals, theta_1_vals):
    """
    Gera os valores da função de custo J(θ) para uma grade de valores de θ0 e θ1.

    @param X np.ndarray: Matriz de características (incluindo o termo de bias).
    @param y np.ndarray: Vetor de valores de saída.
    @param theta_0_vals np.ndarray: Valores para o parâmetro θ0.
    @param theta_1_vals np.ndarray: Valores para o parâmetro θ1.
    @return J_vals np.ndarray: Matriz de valores da função de custo J(θ).
    """
    # Inicializa a matriz de valores da função de custo
    J_vals = np.zeros((len(theta_0_vals), len(theta_1_vals)))

    # Calcula J(θ) para cada par de valores (θ0, θ1)
    for i, t0 in enumerate(theta_0_vals):
        for j, t1 in enumerate(theta_1_vals):
            theta = np.array([t0, t1])
            J_vals[i, j] = compute_cost(X, y, theta)

    return J_vals

def main():
    # Carregando os dados (X, y)
    data = np.loadtxt('Data/ex1data1.txt', delimiter=',')  # Substitua o caminho conforme necessário
    X = data[:, 0]  # Variável independente (população)
    y = data[:, 1]  # Variável dependente (lucro)

    # Adiciona uma coluna de 1s para o termo de bias (X0)
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]  # Agora X_b tem a forma (m, 2)

    # Definindo a faixa de valores para θ0 e θ1
    theta_0_vals = np.linspace(-10, 10, 100)
    theta_1_vals = np.linspace(-1, 4, 100)

    # Gerando os valores da função de custo J(θ) para a grade de valores de θ0 e θ1
    J_vals = generate_contour_data(X_b, y, theta_0_vals, theta_1_vals)

    # Plotando o gráfico de contorno e salvando a figura
    plot_contour(X_b, y, theta_0_vals, theta_1_vals, J_vals)

if __name__ == "__main__":
    main()
