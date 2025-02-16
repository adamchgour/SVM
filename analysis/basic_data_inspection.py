from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class BasicDataInspection(ABC):
    @abstractmethod
    def inspect(self):
        pass
    
class DataInspection(BasicDataInspection):
    def __init__(self, data):
        self.data = data
        
    def inspect(self):
        print(self.data.describe())
        print(self.data.info())
        
class StatisticalInspection(BasicDataInspection):
    '''This class is used to inspect the data using basic statistical methods'''
    def __init__(self, data):
        self.data = data
    def inspect(self):
        print(self.data.describe())
        self.data.hist()
        plt.show()