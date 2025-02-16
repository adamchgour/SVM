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
    
class DataInspectionWithPlot(BasicDataInspection):
    def __init__(self, data):
        self.data = data
        
    def inspect(self):
        self.data.hist()
        plt.show()
class StatisticalInspection(BasicDataInspection):
    def __init__(self, data):
        self.data = data
    def inspect(self):
        print(self.data.describe())
        self.data.hist()
        plt.show()