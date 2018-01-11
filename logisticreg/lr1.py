# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:00:01 2017

@author: Tan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_data = pd.read_csv('data.txt',names=['Exam 1', 'Exam 2', 'Admitted'])

X = my_data.iloc[:,0:2]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
y = my_data.iloc[:,2:3].values
theta = np.zeros([1,3])
alpha = 0.0001
iters = 10000

#sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

#cost function
def cost(X,y,theta):
    first = np.multiply(-y,np.log(sigmoid(X @ theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X @ theta.T)))
    return np.sum(first-second)/len(X)

def gradient(X,y,theta,alpha,iters):
    for i in range(iters):
        theta = theta - ((alpha/len(X)) * np.sum(X * (sigmoid(X @ theta.T)-y)))
    return theta

g = gradient(X,y,theta,alpha,iters)
costNew = cost(X,y,g)

import scipy.optimize as opt  
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))  
cost(result[0], X, y)  






