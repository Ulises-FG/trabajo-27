
# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class LinearRegression():
    def __init__(self):
        pass
    def logistic(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit (self,X,y,epochs=1000,L=0.1,bias=True):
        n = int(len(X)) # Number of elements in X
        y=np.reshape(y,(n,1))
        bias=True
        if bias:
            m=X.shape[1]+1 
            ax = np.ones((n,1))
            X=np.concatenate((X,ax),axis=1)
        else:
            m=X.shape[1]
        thetas= np.zeros((m,1)) # Initial values of thetas

        errores=[]
        iter_=[]
        for i in range(epochs): 
            Y_pred = np.dot(X,thetas)  # The current predicted value of Y
            h=self.logistic(Y_pred)
            error=(1/n)*np.dot(X.T, (h-y))
            thetas-= L*error
            iter_.append(i)
            errores.append(self.mean_squared_error(y,h))
        print(thetas)
        return (iter_,errores)

    def mean_squared_error(self,actual,predicted):
        n = len(actual)
        loss = (-1/n) * (np.dot(actual.T, np.log(predicted)) - np.dot((1-actual).T, np.log(1-predicted)))
        return loss[0][0]
    # Performing Gradient Descent 






