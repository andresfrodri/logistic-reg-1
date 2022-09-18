import numpy as np

#Adding some data to make the prediction later on
N = 100 #100 Samples
D = 2 #Dimension

#In a matrix:
X = np.random.randn(N,D)

#Adding the bias term
ones=np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

#Randomly initialize the weight vector

w = np.random.randn(D+1)

z = Xb.dot(w)

#Let's apply the sigmoid

def sigmoid(z):
    return 1/(1+np.exp(-z))

print(sigmoid(z))