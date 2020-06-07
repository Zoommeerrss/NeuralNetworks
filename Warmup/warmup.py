# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:47:08 2020

@author: Zoommeerrss


    Considere:
    ----------

        X = input matrix
        w = weight vector, as:
            - w[1:] vector without the zero position
            - w[0] vector in the zero position as bias

"""
# use numpy api to make it easy
import numpy as np

# matrix
X = np.array([[1, 2, 3], [4, 5, 6]])
print("X: %s" %(X))


# weight vector and bias
w = np.array([1, 0.1, 0.2, 0.2])
print("w: %s" %(w))

# 1. theta(z)=sum(X.W')+bias = 5.7 for all of them, but pay attention, cuz the cone lines has different ways to get a matrix times vetor product!
zDXW = np.sum(np.dot(X, w[1:]) + w[0]) 
print("z: %s" %(zDXW))

# 2.
zDWXT = np.sum(np.dot(w[1:], X.T) + w[0]) 
print("zm: %s" %(zDWXT))

# 3.
zXDW =  np.sum(X.dot(w[1:]) + w[0]) 
print("zt: %s" %(zXDW))

# 4. the output is a vector, but the sum of the vector elements is 5.7!
zWDXT =  w[1:].dot(X.T) + w[0] 
print("ztm: %s" %(zWDXT))
print("ztm soma: %s" %(np.sum(zWDXT)))

# 5. the iteractive calculus uses ztemp to show the vector positions before the sum # but zf sums each iteration on ztemp field bringin 5.7 at the and similar the samples showed before!

zSum = 0

for xi in zip(X):

    ztemp = np.dot(xi, w[1:]) + w[0]
    zSum += ztemp
    print("xi: %s, w1 %s, w0 %s" %(xi, w[1:].T, w[0]))
    print("loop ztemp: %s" %(ztemp))
    print("loop zf: %s" %(zSum))

print("zf: %s" %(zSum))