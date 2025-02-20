from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import opti 


#Implement Optimization methods in order to maximize the distance between the support vectors
#We first do a One vs One classification and then a One vs All classification
#We will start by the case wehere the data is linearly separable and then implement the kernel trick for non-linearly separable data

def SVM_OvO(X_train,y_train): # TO IMPLEMENT
    def is_linearly_separable(X, y):
        ''' Check if the given data is linearly separable '''
        return np.all(np.dot(y * X, X.T) > 0)
    
    def one_vs_one(X, y, i, j):
        ''' Create a binary dataset for the given classes '''
        mask = (y == i) | (y == j)
        X = X[mask]
        y = y[mask]
        y = np.where(y == i, -1, 1)
        return X, y
    def train(X,y): # There might be an error to check
        Optimizer = opti.QuadraticProgrammingOptimization()
        ''' Train the SVM model '''
        if not is_linearly_separable(X, y):
            raise ValueError("Data is not linearly separable")
        
        n = X.shape[0]
        K = np.dot(X, X.T)
        y = np.where(y == 0, -1, 1)
        astar = Optimizer.optimize(K, y)
        
        # One vs One classification
        phi = np.zeros((n, n))
        for i in range(10):
            for j in range(i + 1, 10):
                X_ij, y_ij = one_vs_one(X, y, i, j)
                phi_ij = Optimizer.optimize(np.dot(X_ij, X_ij.T), y_ij)
                phi += np.outer(y_ij, y_ij) * phi_ij
        decision_value = np.sum([phi[i] * y_train[i] * np.dot(X_train[i], X) for i in range(n)]) + astar
        return decision_value
    return 
