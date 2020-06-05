# NeuralNetworks is Fun!

Welcome to my repository and thank you for see this content!

ML is a powerful and amazing technique to build a lot of programs who will implement learning and get some datasets to predict a bunch of situation
like Weather Forecast, probabilities about a patient develop a desease in certain period of him life, learn words, spell them and more.

Here I'll try to explain for student and newcommers in this context the main concepts and how to apply it in programming language using Python.

So, if you are looking for fun with algorithms, you are in the right place!

Enjoy your time!

# Mathematics Fundamentals for ML

You already know that ML have a big APIs portifolio to help you to start a project with your codes very fast, but, I think the most important part on learning about ML algorithms is the comprehention about mathematics fundamental concepts as linear functions, matrix and vector operations involved, partial derivative and more you will need to now if you are intended to learn how are the thing behing the scenes.

It will be really interesting for you and bring a backgroung when you face some mathematics techniques along your jorney on learning ML!

Fantastic, huh?

So, here you are some codes sample you will use frequently when codding using Python to translate mathematics basic and advanced operations from paper to programming language.

# Python!

The code samples below will show you how to build a matrix, a vector and execute the matrix times vector technique:

# ===================================================
import numpy as np



# matrix
X = np.array([[1, 2, 3], [4, 5, 6]])
print("X: %s" %(X))

# weigth vector and bias
w = np.array([1, 0.1, 0.2, 0.2])
print("w: %s" %(w))

# 1.
# X' times w = 5.7 for all of them, but pay attention, cuz the cone lines has different ways to get a matrix times vetor product!
zDXW = np.sum(np.dot(X, w[1:]) + w[0]) 
print("z: %s" %(zDXW))

# 2.
zDWXT = np.sum(np.dot(w[1:], X.T) + w[0]) 
print("zm: %s" %(zDWXT))

# 3.
zXDW =  np.sum(X.dot(w[1:]) + w[0]) 
print("zt: %s" %(zXDW))

# 4.
# the output is a vector, but the sum of the vector elements is 5.7!
zWDXT =  w[1:].dot(X.T) + w[0] 
print("ztm: %s" %(zWDXT))
print("ztm soma: %s" %(np.sum(zWDXT)))

# 5.
zITERADOR = 0

# the iteractive calculus uses ztemp to show the vector positions before the sum
# but zf sums each iteration on ztemp field bringin 5.7 at the and similar the samples showed before!
for xi in zip(X):
    
    ztemp = np.dot(xi, w[1:]) + w[0]
    zITERADOR += ztemp
    print("xi: %s, w1 %s, w0 %s" %(xi, w[1:].T, w[0]))
    print("loop ztemp: %s" %(ztemp))
    print("loop zf: %s" %(zITERADOR))

print("zf: %s" %(zITERADOR))

# ===================================================

Now, if you tried these code samples and could see how funny they are applying mathematics concepts envolving matrix times vector produtc you are ready to try the codes I'll pull here!

Thank you for come!
