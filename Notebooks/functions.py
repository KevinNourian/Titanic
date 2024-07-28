import pandas as pd
import numpy as np

def MissingValues(data):
    
    name_cols=data.columns[data.isna().any()].tolist()
    missing_values=pd.DataFrame(data[name_cols].isna().sum(), columns=['Number_missing'])
    missing_values['Percentage_missing']=np.round(100*missing_values['Number_missing']/len(data),2)
    
    return missing_values


def Duplicates(data):

    print(f'Duplicates: {data.duplicated().sum()}, ({np.round(100*data.duplicated().sum()/len(data),1)}%)')


def Outliers(data):
    
    numeric_data = data.select_dtypes(include=['number'])

    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)

    IQR = Q3 - Q1

    outliers = (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))

    outlier_counts = outliers.sum()
    print (outlier_counts)