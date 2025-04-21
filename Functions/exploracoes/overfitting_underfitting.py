import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Carregar os dados
data = np.loadtxt('Data/ex1data1.txt', delimiter=',')  # Substitua o caminho conforme necessário
X = data[:, 0].reshape(-1, 1)  # Variável independente (população)
y = data[:, 1]  # Variável dependente (lucro)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Função para realizar a regressão polinomial com diferentes graus
def plot_overfitting_underfitting(X_train, X_test, y_train, y_test, degrees):
    plt.figure(figsize=(12, 8))

    for degree in degrees:
        # Transformar as características em polinômios de grau 'degree'
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Ajustar o modelo de regressão linear
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Previsões
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)

        # Calcular os erros (mean squared error)
        train_error = mean_squared_error(y_train, y_train_pred)
        test_error = mean_squared_error(y_test, y_test_pred)

        # Plotar o modelo ajustado
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_range_poly = poly.transform(X_range)
        y_range_pred = model.predict(X_range_poly)

        # Plotando a curva de regressão para este grau de polinômio
        plt.subplot(2, 2, degrees.index(degree) + 1)
        plt.scatter(X_train, y_train, color='blue', label='Dados de Treinamento')
        plt.scatter(X_test, y_test, color='green', label='Dados de Teste')
        plt.plot(X_range, y_range_pred, label=f'Regressão Polinomial Grau {degree}')
        plt.title(f'Degree {degree} - Train Error: {train_error:.2f}, Test Error: {test_error:.2f}')
        plt.xlabel('População da Cidade')
        plt.ylabel('Lucro')
        plt.legend()
        plt.grid(True)

        # Salvando a figura na pasta Figures
        plt.savefig(f'Figures/curva_aprendizado_grau_{degree}.png')

    plt.tight_layout()
    plt.show()

# Definindo diferentes graus de polinômios para explorar
degrees = [1, 3, 6, 12]  # Grau baixo (underfitting), intermediário (ideal), alto (overfitting)

# Gerar os gráficos de overfitting e underfitting
plot_overfitting_underfitting(X_train, X_test, y_train, y_test, degrees)
