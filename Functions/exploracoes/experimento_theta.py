import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradient_descent import gradient_descent
from compute_cost import compute_cost
from plot_data import load_data

# Inicializar parâmetros
theta_init = np.zeros(2)
alpha = 0.01
num_iters = 1500

# Carregar dados
X, y = load_data()
m = y.size
X_b = np.c_[np.ones((m, 1)), X]  # Adiciona uma coluna de 1s para o termo de bias

theta, J_history, _ = gradient_descent(X_b, y, theta_init.copy(), alpha, num_iters)

# Visualizar o custo
plt.figure()
plt.plot(np.arange(num_iters), J_history, label=f'α = {alpha}')
plt.xlabel('Iterações')
plt.ylabel('Custo J(θ)')
plt.title(f'Convergência da Função de Custo com α = {alpha}')
plt.legend()
plt.grid(True)

# Salvar a figura na pasta Figures
plt.savefig('Figures/fig_convergencia_theta.png')
plt.show()
