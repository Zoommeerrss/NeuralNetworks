# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:20:00 2020

@author: Emerzoom

Escolhendo Algoritmos de Classificacao e Realizando Pre-Processamento no Dataset

    -escolher um algoritmo para classificar
    -aprender a pre-processar o dataset para otimizar a performance do algoritmo

"""

# # trata datasets já embutidos
# from sklearn import datasets
# # separa e prepara o treinamento e o teste
# from sklearn.model_selection import train_test_split
# # Padroniza o dataset pre-processando automaticamente
# from sklearn.preprocessing import StandardScaler
# # medir a acuracia do treinamento
# from sklearn.metrics import accuracy_score
# # sci-fi-kit Perceptron
# from sklearn.linear_model import Perceptron

# # importar o perceptron implementado anteriormente
# import Perceptron_iris_aula as perceptron

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap


"""
    Carregar o dataset
""" 
def load_dataset():
    
    df = pd.read_csv('iris_perceptron_samples.data', header=None)
    df.tail()
    
    # pegar setosa e versicolor
    y = df.iloc[0:100, 4].values
    
    y = np.where(y == 'Iris-setosa', 0, 1)
    print("y %s" %y)
    
    # pegar tamanho das petalas e das sepalas
    # X = df.iloc[0:100, [0, 1, 2, 3]].values
    X = df.iloc[0:100, [0, 2]].values
    print("X %s" %X)
    
    
    
    # plot primeiros 50
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    # plot 100 restantes
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')
    
    plt.xlabel('sepala [cm]')
    plt.ylabel('petala [cm]')
    plt.legend(loc='upper left')
    
    # plt.savefig('images/02_06.png', dpi=300)
    plt.show()
    
    return X, y


"""
    Traca a linha como fronteira de decisao utilizando cores para separar as classes
""" 
def plot_decision_regions(X, y, classifier=None, test_idx=None, resolution=0.02):
        
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
       
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = predict(np.array([xx1.ravel(), xx2.ravel()]).T)      
    Z = Z.reshape(xx1.shape)
        
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')   
        
    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        
        plt.scatter(X_test[:, 0], 
                    X_test[:, 1], 
                    c='', 
                    edgecolor='black', 
                    alpha=1.0, 
                    linewidth=1, 
                    marker='o', 
                    s=100, 
                    label='test set')
        
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.2)
    plt.show()

   
"""
    Predicao do modelo
"""
def predict(xi):
        
    return np.where(calcule(xi) > 0.0, 1, 0)  

"""
    Calcula a matriz vezes o peso
"""
def calcule(xi):
        
    return np.dot(xi, w[1:].T) + w[0] 
    
    
"""
    Tabelas AND, OR, XOR
"""    

X, y = load_dataset()

print(X)   

print(y)

random_state = 1
rgen = np.random.RandomState(random_state)
# vetor de pesos 
w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
print(w)

erro = 0
erros = []
eta = 0.1
epocaMax = 1000
it = 0
eqm = 0
eqmMax = 1e-2

"""
    Percorre as epocas do algoritmo
"""
while it < epocaMax or eqm > eqmMax:    
    
    erro = 0

    for xi, yi in zip(X, y):
                
        pred = predict(xi)
        
        update = eta * (yi - pred)
        w[1:] += update * xi
        w[0] += update
        #erro += int(update != 0.0)
        #erro += update      
        
        # Min Square Error 
        eqm = (update**2).sum() / 2.0
        
        erros.append(eqm)
        print("eqm %s" %eqm) 
        
    it = it + 1
    

# plot primeiros 50
# plt.scatter(X[:1, 0], X[:1, 1],
#             color='blue', marker='x', label='UM')
# # plot 100 restantes
# plt.scatter(X[1:, 0], X[1:, 1],
#             color='red', marker='o', label='ZERO')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# # plt.savefig('images/02_06.png', dpi=300)
# plt.show()
    
# separar regioes   
plot_decision_regions(X, y)    

# erros
epocas = np.arange(1, len(erros) + 1)
plt.plot(epocas, erros)
plt.xlabel('Epocas')
plt.ylabel('Erros')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.2)
plt.grid()
plt.show()