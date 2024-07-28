import pandas as pd
import numpy as np

def MissingValues(data):
    
    name_cols=data.columns[data.isna().any()].tolist()
    missing_values=pd.DataFrame(data[name_cols].isna().sum(), columns=['Number_missing'])
    missing_values['Percentage_missing']=np.round(100*missing_values['Number_missing']/len(data),2)
    
    return missing_values


def DuplicateValues(data):

    print(f'Duplicates: {data.duplicated().sum()}, ({np.round(100*data.duplicated().sum()/len(data),1)}%)')