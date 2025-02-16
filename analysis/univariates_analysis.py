from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class UnivaratesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        ''' Perform a specific type of univariates analysis on the given DataFrame '''
        self.identify_univariates(df)
        self.plot_univariates(df)
    
    @abstractmethod 
    def identify_univariates(self, df: pd.DataFrame):
        ''' Identifies univariates in the given DataFrame '''
        pass
    def plot_univariates(self, df: pd.DataFrame):
        ''' Plots the univariates in the given DataFrame '''
        pass

class SimplUnivariatesAnalyser(UnivaratesAnalysisTemplate):
    def identify_univariates(self, df: pd.DataFrame):
        ''' Identify univariates in the given DataFrame '''
        univariates = df.describe().T
        print("Univariates:")
        print(univariates)

    def plot_univariates(self, df: pd.DataFrame):
        ''' Plot univariates in the given DataFrame '''
        plt.figure(figsize=(12, 6))
        plt.title("Univariates")
        sns.histplot(df, kde=True)
        plt.show()