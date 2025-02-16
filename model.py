from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        ''' Fit the model to the given data '''
        pass
    
    @abstractmethod
    def predict(self, X):
        ''' Predict the target variable for the given data '''
        pass

class SupportVectorMachine(Model):
    NotImplemented
