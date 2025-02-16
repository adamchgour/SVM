from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MultivariatesAnalysis(ABC):
    @abstractmethod
    def analyze(self, df):
        ''' Perform a specific type of multivariates analysis on the given DataFrame '''
        pass

class PairplotAnalysis(MultivariatesAnalysis):
    def analyze(self, df):
        '''plots the relationship between all numerical features'''
        
        plt.figure(figsize=(12, 6))
        sns.pairplot(df)
        plt.show()

class HeatmapAnalysis(MultivariatesAnalysis):
    def analyze(self, df):
        '''plots the relationship between all features'''
        
        plt.figure(figsize=(12, 6))
        plt.title("Relationship between all features")
        sns.heatmap(df.select_dtypes(include=['int64', 'float64']).corr(), annot=True, cmap='coolwarm')
        plt.show()

class ScatterMatrixAnalysis(MultivariatesAnalysis):
    def analyze(self, df):
        '''plots the relationship between all numerical features'''
        
        plt.figure(figsize=(12, 6))
        plt.title("Relationship between all numerical features")
        sns.scatter_matrix(df, alpha=0.2)
        plt.show()