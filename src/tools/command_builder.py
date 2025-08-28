import pandas as pd
from . import commands
from . import types_
class Builder:
    
    def __init__(self,df: pd.DataFrame):
        self.df = df
    
    def build(self,command,parameters):
        command = command[:command.rfind("_")]
        for i in range(len(parameters)):
            if isinstance(parameters[i],str):
                parameters[i] = parameters[i][:parameters[i].rfind("_")]

        if '"' in command:
            command = command[1:-1]
        
        for i in range(len(parameters)):
            if isinstance(parameters[i],str) and '"' in parameters[i]:
                parameters[i] = parameters[i][1:-1]

        match command:
            case 'smooth':
                return commands.smooth(*parameters)
            case 'trend':
                return commands.trend(*parameters)
            case 'seasonal':
                return commands.seasonal(*parameters)
            case 'acf':
                return commands.acf(*parameters)
            case 'pacf':
                return commands.pacf(*parameters)
            case 'forecast':
                return commands.forecast(*parameters)
            case 'correlation':
                return commands.correlation(*parameters)
            case 'kl_divergence':
                return commands.kl_divergence(*parameters)
            case 'entropy':
                return commands.entropy(*parameters)
            case 'pca':
                return commands.pca(*parameters)
            case 'mean':
                return commands.mean(*parameters)
            case 'std':
                return commands.std(*parameters)
            case 'plot':
                return commands.plot(*parameters)
            case 'line':
                if isinstance(parameters[0],(types_.Lines,types_.Points,types_.Distribution)):
                    return commands.line(*parameters)
                return commands.line(self.df,*parameters)
            case 'points':
                if isinstance(parameters[0],(types_.Lines,types_.Points,types_.Distribution)):
                    return commands.points(*parameters)
                return commands.points(self.df,*parameters)
            case 'dist':
                if isinstance(parameters[0],(types_.Lines,types_.Points,types_.Distribution)):
                    return commands.dist(*parameters)
                return commands.dist(self.df,*parameters)