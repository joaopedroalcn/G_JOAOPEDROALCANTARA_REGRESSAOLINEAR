import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradient_descent import gradient_descent
from compute_cost import compute_cost
from plot_data import load_data


alphas = [0.0001, 0.001, 0.01]
num_iters = 1500
theta_init = np.zeros(2)

X, y = load_data()  # Agora load_data retorna X e y
m = y.size
# Normalização dos dados (opcional, mas recomendada)
X_normalized = (X - np.mean(X)) / np.std(X)  # Normaliza X
X_b = np.c_[np.ones((m, 1)), X_normalized]  # Adiciona uma coluna de 1s para o termo de bias

plt.figure()
for alpha in alphas:
    theta, J_history, _ = gradient_descent(X_b, y, theta_init.copy(), alpha, num_iters)
    plt.plot(np.arange(num_iters), J_history, label=f'α = {alpha}')
plt.xlabel('Iterações')
plt.ylabel('Custo J(θ)')
plt.title('Convergência da Função de Custo para Diferentes α')
plt.legend()
plt.grid(True)

# Salvar a figura na pasta Figures
plt.savefig('Figures/fig_convergencia_alpha.png')
plt.show()
