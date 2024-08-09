import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def MissingValues(data):
    
    name_cols=data.columns[data.isna().any()].tolist()
    missing_values=pd.DataFrame(data[name_cols].isna().sum(), columns=['NumberMissing'])
    missing_values['PercentageMissing']=np.round(100*missing_values['NumberMissing']/len(data),2)
    
    return missing_values



def UniqueValues(data):
    
    categorical_columns_list = data.select_dtypes(include=['object']).columns.tolist()
    unique_values_dic = {}

    for column in categorical_columns_list:
        unique_values_dic[column] = data[column].nunique()

    for column, values in unique_values_dic.items():
        print(f'Unique Values in {column}: {values}')



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



def side_by_side_countplot(data_1, data_2, title_1, title_2, labels, feature, palette):

    plt.rcParams.update(params)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))  


    sns.countplot(
        ax=ax[0],  
        x="HomePlanet",
        hue="Transported",
        data=train,
        palette=[color_1, color_2]
    )
    ax[0].set_xlabel("Home Planet")
    ax[0].set_title("Transported by Home Planet", fontsize=size)
    ax[0].legend(title='Transported', loc='upper right')


    sns.countplot(
        ax=ax[1], 
        x="Destination",
        hue="Transported",
        data=train,
        palette=[color_1, color_2]
    )
    ax[1].set_xlabel("Destination Planet")
    ax[1].set_title("Transported by Destination Planet", fontsize=size)
    ax[1].legend(title='Transported', loc='upper right')

    plt.tight_layout()

    plt.show()
   
    
def passenger_distribution(data, feature, Boolean):
    
    transported = data[data['Transported'] == Boolean]
    transported_feature = transported[feature].value_counts()

    transported_feature_true = transported_feature.get(1, 0)
    transported_feature_false = transported_feature.get(0, 0)

    return transported_feature_true, transported_feature_false