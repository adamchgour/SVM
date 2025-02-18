from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplit(ABC):
    @abstractmethod
    def split(self, data):
        pass

class TrainTestSplit(DataSplit):
    def __init__(self, train_size):
        self.train_size = train_size
    def split(self, data):
        return train_test_split(data, train_size=self.train_size)

