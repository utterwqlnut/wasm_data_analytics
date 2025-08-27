import pandas as pd
import numpy as np
# Generic Types for Analysis: Points, Lines, Distribution Meant to act as Enums

class Points:
    # Just a simple wrapper class keeps the row structure
    def __init__(self, df: pd.DataFrame):
        self.points = df

    def get_points(self):
        return self.points
    
    def set_points(self,df: pd.DataFrame):
        self.points = df

class Lines:
    # Another simple wrapper class
    def __init__(self,points):
        if isinstance(points,Points):
            self.df = points.get_points()
        elif isinstance(points,pd.DataFrame):
            self.df = points
        else:
            raise TypeError("Line expectes points or dataframe as parameter")
    
    def get_line(self):
        return self.df
    
    def set_line(self,df:pd.DataFrame):
        self.df=df
        
class Distribution:
    # Another simple wrapper class
    def __init__(self,points,bins):
        if isinstance(points,Points):
            self.df = points.get_points()
        elif isinstance(points,pd.DataFrame):
            self.df = points
        else:
            raise TypeError("Line expectes points or dataframe as parameter")
        
        data = self.df.values

    def get_distribution(self):
            return self.df
        
    def set_distribution(self,df:pd.DataFrame):
        self.df=df 