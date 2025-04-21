import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Função de custo para regressão linear
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    J = (1 / (2 * m)) * np.sum(errors ** 2)
    return J

# Função de descida do gradiente
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta = theta - alpha * gradient
        J_history[i] = compute_cost(X, y, theta)
    
    return theta, J_history

# Função para calcular o erro de treinamento e validação
def learning_curve(X_train, y_train, X_val, y_val, alpha, num_iters):
    m_train = len(y_train)
    training_errors = []
    validation_errors = []
    
    # Iterando sobre diferentes tamanhos de conjunto de treinamento
    for i in range(1, m_train + 1):
        # Treinando o modelo com os primeiros i exemplos
        X_train_i = X_train[:i]
        y_train_i = y_train[:i]
        
        # Adiciona a coluna de 1s para o termo de bias
        X_train_i = np.c_[np.ones((X_train_i.shape[0], 1)), X_train_i]
        
        # Inicializa os parâmetros
        theta_init = np.zeros(X_train_i.shape[1])
        
        # Treinando o modelo com o subconjunto de dados
        theta, _ = gradient_descent(X_train_i, y_train_i, theta_init, alpha, num_iters)
        
        # Calculando os erros de treinamento e validação
        train_error = compute_cost(X_train_i, y_train_i, theta)
        val_error = compute_cost(np.c_[np.ones((X_val.shape[0], 1)), X_val], y_val, theta)
        
        # Armazenando os erros
        training_errors.append(train_error)
        validation_errors.append(val_error)
    
    return training_errors, validation_errors

def main():
    # Carregar os dados
    data = np.loadtxt('Data/ex1data1.txt', delimiter=',')  # Substitua pelo caminho do seu arquivo
    X = data[:, 0]  # Variável independente (população)
    y = data[:, 1]  # Variável dependente (lucro)
    
    # Dividindo os dados em treinamento e validação (80% para treino, 20% para validação)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Adicionando uma coluna de 1s para o termo de bias
    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_val = np.c_[np.ones((X_val.shape[0], 1)), X_val]
    
    # Parâmetros
    alpha = 0.01  # Taxa de aprendizado
    num_iters = 1500  # Número de iterações
    
    # Calculando a curva de aprendizado
    training_errors, validation_errors = learning_curve(X_train, y_train, X_val, y_val, alpha, num_iters)
    
    # Plotando as curvas de aprendizado
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(training_errors) + 1), training_errors, label='Erro de Treinamento')
    plt.plot(range(1, len(validation_errors) + 1), validation_errors, label='Erro de Validação')
    plt.xlabel('Número de Exemplos de Treinamento')
    plt.ylabel('Erro')
    plt.title('Curvas de Aprendizado')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figures/curvas_de_aprendizado.png')  # Salvar figura
    plt.show()

if __name__ == "__main__":
    main()
