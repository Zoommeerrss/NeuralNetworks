# -*- coding: utf-8 -*-
"""


@author: Zoommeerrss

Sigmoid and Derivative


"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))   

def sigmoid_derivative(x):
    
    s = sigmoid(x)
    return s * (1 - s)

# linespace generate an array from start and stop value, 100 elements
values = np.linspace(-50, 50, 500)

# prepare the plot, associate the color r(ed) or b(lue) and the label 
plt.plot(values, sigmoid(values), 'r')
plt.plot(values, sigmoid_derivative(values), 'b')

# Draw the grid line in background.
plt.grid()

# Title & Subtitle
plt.title('Sigmoid and Sigmoid derivative functions')

# plt.plot(x)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# create the graph
plt.show()