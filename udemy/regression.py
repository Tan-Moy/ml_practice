# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:20:04 2017

@author: Tan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('housing.data',delim_whitespace=True,header=None,names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]) #instead of delim_whitespace we can use sep='\s+' for separating whitespaces

##################################################
#1.EDA: Exploratory data analysis
##################################################

df.describe() # describes the data
sns.pairplot(df,size=1.5) #create a pair plot

##################################################
#2.Correlation analysis and features selection
##################################################

df.corr() # gives a correlation chart then we can choose features with high correlations
 #we can also plot a heatmap that shows correlation
plt.figure(figsize=(16,10))
sns.heatmap(df.corr(),annot=True)
plt.show()

##################################################
#3.Linear Regression with scikit learn
##################################################
X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X,y)

model.coef_
model.intercept_

plt.figure(figsize=(12,10))
sns.regplot(X,y)
plt.xlabel('average number of rooms per dwelling')
plt.ylabel('Median value of owner occupied homes in $1000')
plt.show()

sns.jointplot(x='MEDV',y='RM',data=df,kind='reg',size=10)
plt.show()

##################################################
X1 = df['LSTAT'].values.reshape(-1,1)
y1 = df['MEDV'].values

model1 = LinearRegression()
model1.fit(X1,y1)

plt.figure(figsize=(12,10))
sns.regplot(X1,y1)
# we can see that the model is not a good fit for the data. So we use robust regression. Ransac below is one of them

##################################################
#4.Ransac Algorithm
##################################################
X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor()

ransac.fit(X,y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_x = np.arange(3,10,1)
line_y_ransac = ransac.predict(line_x.reshape(-1,1))

sns.set(style='darkgrid')
plt.scatter(X[inlier_mask],y[inlier_mask],
            c='blue',marker='s',label='inliers')
plt.scatter(X[outlier_mask],y[outlier_mask],
            c='brown',marker='s',label='outliers')
plt.plot(line_x,line_y_ransac,color='red')
plt.xlabel('avg number of rooms per dwelling')
plt.ylabel('median values of owner occupied homes in 1000 dollars')
plt.legend(loc='upper right')

ransac.estimator_.coef_
ransac.estimator_.intercept_


##################################################
#5.Performance Evaluation of regression model
##################################################
#we should split data into training set(60%),evaluation set(20%), and test set(20%).

from sklearn.model_selection import train_test_split
X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
lr = LinearRegression()
lr.fit(X_train,y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

#5.1.Residual analysis:comparing the in sample error with out sample error
#############################################################################
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'blue', marker='o',label="training data") #insample error 
plt.scatter(y_test_pred, y_test_pred - y_test, c = 'orange', marker='*',label="test data") #out of sample error
plt.xlabel('Predicted values')
plt.ylabel('residuals')
plt.legend(loc="upper left")
plt.hlines(y=0,xmin=-10, xmax=50, lw=2, color='k')
plt.xlim = ([-10,50])

#5.2.Mean squared error
########################
from sklearn.metrics import mean_squared_error
mean_squared_error(y_train,y_train_pred)
mean_squared_error(y_test,y_test_pred)
#we can see that out of sample error is greater than insample error

#5.3.Co-efficient of determination(R^2)
########################################
from sklearn.metrics import r2_score
r2_score(y_train,y_train_pred)
r2_score(y_test,y_test_pred)
#insample has a greater r-squared value than out of sample




