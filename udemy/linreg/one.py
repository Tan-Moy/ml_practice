# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:52:04 2017

@author: Tan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = 10 * np.random.rand(100)
y = 3 * x + np.random.randn(100)

plt.scatter(x,y)

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept="true")

X = x.reshape(-1,1) #convert into a matrix with 1 coloumn. Python figures out the rows

model.fit(X,y) #fit the model i.e apply the learning algorithm

model.intercept_ #gives the intercept(theta1) after learning is complete
model.coef_ #gives the slope(theta2) after learning is complete


x_fit = np.linspace(-1,11) # creates a array of evenly spaced numbers
x_fit = x_fit.reshape(-1,1) # creates a matrix from above array
y_fit = model.predict(x_fit) # predict the value of y matrix using the intercept and coef values found earlier

plt.scatter(x,y) #scatter plot for x and y
plt.plot(x_fit,y_fit) #create a line plot using x_fit and y_fit i.e put a point on the graph for each corresponding x_fit and y_fit values