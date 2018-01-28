import numpy as np
import pandas as pd

N = 100
D = 2

X = np.random.randn(N,D)
ones = np.array([[1] * N]).T
Xb = np.concatenate((ones,X),axis=1)

w = np.random.randn(D + 1,1)

z = Xb.dot(w) # according to doc .dot() does matrix multiplication for 2d arrays

def sigmoid(z):
	return 1/(1 + np.exp(-z))

print (sigmoid(z))

df = pd.read_csv('ecommerce_data.csv')

