# -*- coding: utf-8 -*-
"""


@author: Zoommeerrss


Min Square Technique



"""


import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline


# vetor Xzao
#X = 2 * np.random.rand(100, 1)
X = np.array([1, 1, 2, 3, 4, 4, 4])

# vetor y
#y = 4 + 3 * X + np.random.randn(100, 1)
y = np.array([1, 4, 3, 1, 2, 3, 5])

X_total = len(X)

X_bias = np.c_[np.ones((X_total, 1)), X]

theta_best = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)

print("theta_best: %s " %(theta_best))

sumX = 0
sumY = 0
sumXQ = 0
sumXY = 0

n = len(X)

for valX, valY in zip(X, y):
    
    sumX += valX
    sumY += valY
    sumXQ += valX**2
    sumXY += valX * valY
    

print("somatorios: %s, %s, %s, %s" %(sumX, sumY, sumXQ, sumXY))

a = ((sumY * sumXQ) - (sumX * sumXY)) / ((n * sumXQ) - (sumXQ**2))

b = ((n * sumXY) - (sumX * sumY)) / ((n * sumXQ) - (sumXQ**2))

print("a: %2.5f e b: %2.5f" %(a, b))


# yFinal = a * X + b
yFinal = theta_best[1] * X + theta_best[0]

print("coeficientes: %s, %s" %(a, b))

plt.plot(X, yFinal, 'r')
plt.scatter(X, y)
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18) 
_ = plt.axis([-1, 5, -1, 5])


plt.grid()
plt.show()