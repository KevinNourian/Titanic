import pandas as pd
import numpy as np

# Missing Values
def MissingValues(data):
    
    name_cols=data.columns[data.isna().any()].tolist()
    missing_values=pd.DataFrame(data[name_cols].isna().sum(), columns=['NumberMissing'])
    missing_values['PercentageMissing']=np.round(100*missing_values['NumberMissing']/len(data),2)
    
    return missing_values


# Unique Values
def UniqueValues(data):
    
    categorical_columns_list = data.select_dtypes(include=['object']).columns.tolist()
    unique_values_dic = {}

    for column in categorical_columns_list:
        unique_values_dic[column] = data[column].nunique()

    for column, values in unique_values_dic.items():
        print(f'Unique Values in {column}: {values}')


# Duplicates
def Duplicates(data):

    print(f'Duplicates: {data.duplicated().sum()}, ({np.round(100*data.duplicated().sum()/len(data),1)}%)')


# Outliers
def Outliers(data):
    
    numeric_data = data.select_dtypes(include=['number'])

    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)

    IQR = Q3 - Q1

    outliers = (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))

    outlier_counts = outliers.sum()
    print (outlier_counts)


# Side-by-Side Bar Plots
def side_by_side_barplot(data_1, data_2, title_1, title_2, labels, feature, y, palette):

    '''
    Creates a side-by-side bar plot comparing two datasets.
    '''

    plt.rcParams.update(params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.barplot(data=data_1, x=feature, y=y, ax=ax1, palette=palette)
    sns.barplot(data=data_2, x=feature, y=y, ax=ax2, palette=palette)

    ax1.set_xlabel(feature)
    ax1.set_ylabel(y)
    ax2.set_xlabel(feature)
    ax2.set_ylabel(y)

    total_count1 = data_1[y].sum()
    for container in ax1.containers:
        labels = [f'{(v.get_height() / total_count1 * 100):.1f}%' for v in container]
        ax1.bar_label(container, labels=labels, size=size)

    total_count2 = data_2[y].sum()
    for container in ax2.containers:
        labels = [f'{(v.get_height() / total_count2 * 100):.1f}%' for v in container]
        ax2.bar_label(container, labels=labels, size=size)

    ax1.set_title(title_1)
    ax2.set_title(title_2)

    sns.despine()

    plt.show()    