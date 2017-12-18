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

##################################################
#6.Multiple Linear Regression
##################################################
from sklearn.datasets import load_boston
boston_data = load_boston()

df = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
X = df
y = boston_data.target

#we will be using statsmodels for data exploration
import statsmodels.api as sm
import statsmodels.formula.api as smf

form_lr = smf.ols(formula = 'y ~ CRIM+ZN+INDUS+CHAS+NOX+RM+AGE+DIS+RAD+TAX+PTRATIO+B+LSTAT',data = df)
mlr = form_lr.fit()
mlr.summary()
# RSquared of 0.741 means that 74.1% of the price is due to the all the features taken into account, in this case, all of them.
# Now Rsquare increases with increase in no of independant variable. In that case we use Adj Rsquares to keep it Rsquare in check i.e to keep it from getting to influenced.

#we can idetify key features by:

#6.1.Detecting MultiCollinearity using Eigen Vectors
####################################################
# What is MultiCollinearity and why is it bad?
# https://stats.stackexchange.com/questions/1149/is-there-an-intuitive-explanation-why-multicollinearity-is-a-problem-in-linear-r
# How to detect: https://stackoverflow.com/questions/25676145/capturing-high-multi-collinearity-in-statsmodels

eigenvalues,eigenvectors = np.linalg.eig(df.corr())
pd.Series(eigenvalues).sort_values() #after sorting we can see that 8th value is low and thus shows collinearity. It's corresponding eigen vector will give us the culprit variables
np.abs(pd.Series(eigenvectors[:,8])).sort_values(ascending=False) #we sort the absolute values of the eigenvector of 8th eigenvalue in descending order
# And the culprits with most contribution to 8th are 9,8, and 2
print(df.columns[2],df.columns[8],df.columns[9])
# thus, INDUS RAD TAX are causing the collinearity problem

#6.2.Standardize variables to identify key features
####################################################
# for indepth tutorial: https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/
# after standardizing we can look at the coefficients. Higher the coefficient higher is the corresponding features ability to explain variability
#in this case we find lstat to be the most significant feature

#6.2.Using Rsquared
####################
# Compare rsquare with and without a feature
# A significant change in Rsquare signifies the importance of an feature
from sklearn.metrics import r2_score
linear_reg = smf.ols(formula = 'y ~ CRIM+ZN+INDUS+CHAS+NOX+RM+AGE+DIS+RAD+TAX+PTRATIO+B+LSTAT',data = df)
benchmark = linear_reg.fit()
r2_score(y,benchmark.predict(df)) #0.74060774286494269

#without lstat
linear_reg = smf.ols(formula = 'y ~ CRIM+ZN+INDUS+CHAS+NOX+RM+AGE+DIS+RAD+TAX+PTRATIO+B',data = df)
benchmark = linear_reg.fit()
r2_score(y,benchmark.predict(df))  #0.6839521119105445, significant change

#without age
linear_reg = smf.ols(formula = 'y ~ CRIM+ZN+INDUS+CHAS+NOX+RM+DIS+RAD+TAX+PTRATIO+B+LSTAT',data = df)
benchmark = linear_reg.fit()
r2_score(y,benchmark.predict(df)) #0.74060603879043385, barely any change
