
# coding: utf-8

# In[191]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[207]:

data = pd.read_csv('data2.csv',names=['X','Y'])
data.head()


# In[208]:

data.describe()


# In[209]:

# data.plot(kind='scatter', x='X',y='Y',figsize=(12,8))


# In[210]:

data.insert(0,'Ones',1)
data.head()


# In[211]:

#create the training and target vars
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]


# In[212]:

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0.0,0.0]))


# In[213]:

#print(theta)


# In[214]:

X.shape,theta.shape,y.shape


# In[215]:

def computeCost (X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))


# In[216]:

computeCost(X,y,theta)


# In[217]:

#print(np.sum(np.multiply(2782,X[:,0])))


# In[218]:

def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error,X[:,j]) #differentiating the term inside of sigma in cost function gives this
            temp[0,j] = theta[0,j] - ((alpha/len(X))*np.sum(term)) #from gradient descent formula
        
        theta = temp
        cost[i] = computeCost(X,y,theta)
        
    return theta,cost


# In[223]:

# initialize learning parameters 
alpha = 0.01
iters = 1000

#perform gradient descent 
g, cost = gradientDescent(X, y, theta, alpha, iters)  
print(g)


# In[222]:

computeCost(X, y, g)


# In[ ]:



