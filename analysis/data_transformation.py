from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.random_projection import GaussianRandomProjection

class DataTransformation(ABC):
    @abstractmethod
    def transform(self, df):
        ''' Perform a specific type of data transformation on the given DataFrame '''
        pass

class JohnsonnLinderstraussLemma(DataTransformation): # I stil need to understand how I'm going to visualize data with this and how I can interpret it
    def transform(self, df):
        ''' Reduces the dimensionality of the given DataFrame using the Johnson-Linderstrauss Lemma '''
        transformer = GaussianRandomProjection(n_components=2) #log(4) < 2
        reduced_df = transformer.fit_transform(df)
        return pd.DataFrame(reduced_df, columns=['Component1', 'Component2'])
