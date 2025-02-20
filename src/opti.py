from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Optimization(ABC):
    @abstractmethod
    def optimize(self, X, y,learning_step):
        ''' Optimize the model to maximize the distance between the support vectors '''
        pass

class QuadraticProgrammingOptimization(Optimization):
    def optimize(self, K, y, learning_rate=0.01, max_iter=1000):
        ''' Solve the quadratic programming problem '''
        n = K.shape[0]
        phi = np.zeros(n)
        
        for _ in range(max_iter):
            gradient = 2 * np.dot(K, phi) - np.ones(n)
            phi -= learning_rate * gradient
            
            # Projection step to ensure constraints are satisfied
            phi = np.maximum(phi, 0)  # Ensure phi >= 0
            phi -= np.dot(y, phi) * y / np.dot(y, y)  # Ensure y.T @ phi = 0

        return phi
