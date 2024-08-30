import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt



size = 20
params = {
    "font.family": "Times New Roman",
    "font.size": size,
    "axes.labelsize": size,
    "xtick.labelsize": size * 0.75,
    "ytick.labelsize": size * 0.75,
    "figure.titlesize": size * 1.5,
    "axes.titlesize": size * 1.5,
    "axes.titlepad": size,
    "axes.labelpad": size - 10,
    "lines.linewidth": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "legend.fontsize": size,
    "figure.figsize": (10, 6),
}




def MissingValues(data):

    """
    Calculates the number and percentage of missing values for each column in a DataFrame.
    """
    
    name_cols=data.columns[data.isna().any()].tolist()
    missing_values=pd.DataFrame(data[name_cols].isna().sum(), columns=['NumberMissing'])
    missing_values['PercentageMissing']=np.round(100*missing_values['NumberMissing']/len(data),2)
    
    return missing_values




def UniqueValues(data):

    """
    Prints the number of unique values for each categorical column in the DataFrame.
    """
    
    categorical_columns_list = data.select_dtypes(include=['object']).columns.tolist()
    unique_values_dic = {}

    for column in categorical_columns_list:
        unique_values_dic[column] = data[column].nunique()

    for column, values in unique_values_dic.items():
        print(f'Unique Values in {column}: {values}')




def duplicates(data):

    """
    Prints the number and percentage of duplicate rows in the DataFrame.
    """

    print(f'Duplicates: {data.duplicated().sum()}, ({np.round(100*data.duplicated().sum()/len(data),1)}%)')




def outliers(data):

    """
    Prints the count of outliers in each numerical column of the DataFrame based on the IQR method.
    """

    numeric_data = data.select_dtypes(include=['number'])
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1

    outliers = (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))

    outlier_counts = outliers.sum()
    print (outlier_counts)




def passenger_distribution(data, feature, Boolean):
    transported = data[data['Transported'] == Boolean]
    transported_feature = transported[feature].value_counts()

    transported_feature_true = transported_feature.get(1, 0)
    transported_feature_false = transported_feature.get(0, 0)

    return transported_feature_true, transported_feature_false




def log_transform(data, col):
    """
    Applies a log transformation (log1p) to the specified column in the DataFrame.
    """

    data[col] = np.log1p(data[col])

    return data


def countplot(data, x, hue, palette, order, title, x_label, y_label, legend_title):

    """
    Creates a count plot with customized appearance including axis labels, title, and legend.
    """

    plt.rcParams.update(params)
    plt.figure(figsize=(10, 6))

    sns.countplot(data=data, x=x, hue= hue, palette=palette, order=order)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.legend(title= legend_title)
    plt.tight_layout()

    plt.show()





def sidebyside_countplot(data_1, data_2, feature, title_1, title_2, labels, order_1, order_2, color_1, color_2):

    """
    Plots side-by-side count plots for a specified feature from two datasets using different titles, orders, and colors.
    """

    plt.rcParams.update(params)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    order_1 = data_1[feature].value_counts().index
    sns.countplot(
        ax=ax[0],
        x=feature,
        data=data_1,
        order=order_1,
        palette=[color_1]
    )
    ax[0].set_xlabel(labels)
    ax[0].set_title(title_1, fontsize=size)

    order_2 = data_2[feature].value_counts().index
    sns.countplot(
        ax=ax[1],
        x=feature,
        data=data_2,
        order=order_2,
        palette=[color_2]
    )
    ax[1].set_xlabel(labels)
    ax[1].set_title(title_2, fontsize=size)


    plt.tight_layout()
    plt.show()



def combined_countplot(data_1, data_2, feature, title, order, color_1, color_2, labels, label_fontsize,
                       title_fontsize, hue_order):
    
    """
    Creates a combined count plot with data from two datasets, distinguishing categories using different colors.
    """

    combined_data = pd.concat([data_1, data_2], axis=0)

    plt.rcParams.update(params)
    plt.figure(figsize=(10, 7))


    sns.countplot(
        data=combined_data,
        x=feature,
        hue='Transported',
        order=order,
        palette=[color_1, color_2],
        hue_order = hue_order
    )

    plt.xlabel(labels, fontsize=label_fontsize)
    plt.ylabel('Count', fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()




def sidebyside_barplot(data_1, data_2, title_1, title_2, labels, feature, y, palette):

    """
    Creates a side-by-side bar plot comparing two datasets.
    """

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




def piechart(data, title, colors, labels, size):

    """
    Plots a pie chart with specified data, colors, labels, and title.
    """

    fig, axes = plt.subplots(1, 1, figsize=(10, 7))

    axes.set_title(title, fontsize=size * 1.5, pad=size)
    axes.pie(
        data,
        colors=colors,
        labels=labels,
        startangle=90,
        autopct="%0.2f%%",
        wedgeprops={"edgecolor": "black"},
        textprops={"fontsize": size + 5},
    )

    plt.tight_layout()

    plt.show()



def create_plot_mi_scores(features, mi_scores):
    
    '''
    Creates a plot of mutual information scores.
    '''

    plt.rcParams.update({'figure.autolayout': True}) 

    scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
    scores = scores.sort_values(ascending=False)

    plt.figure(figsize=(15, 6))
    scores.plot(kind="line", marker='o')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.title("Mutual Information Scores")
    plt.xlabel("Feature")
    plt.ylabel("MI Score")

    plt.xticks(ticks=range(len(scores)), labels=scores.index, rotation=45, ha='right')

    plt.tight_layout()

    plt.show()




def create_heatmap(data, title):

    '''
    Creates a Seaborn heatmap.
    '''

    plt.rcParams.update(params)
    corr = data.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(25, 25))

    cmap = sns.diverging_palette(230, 10, as_cmap=True)
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
        cmap=plt.cm.Reds,
    )

    heatmap.set_title(
        title,
        fontdict={"fontsize": size},
        pad=12,
    )
    plt.xlabel("")
    plt.ylabel("")




def sidebyside_piechart(data_1, data_2, title_1, title_2, color_1, color_2, lables, size):

    """
    Plots two side-by-side pie charts with specified titles, colors, and labels.
    """

    plt.rcParams.update(params)
    fig, ax = plt.subplots(1, 2, figsize=(25, 15))

    ax[0].pie(
        data_1,
        startangle=90,
        colors = [color_1, color_2],
        autopct="%0.1f%%",
        wedgeprops={"edgecolor": "black"},
        textprops={"fontsize": size*1.5},
        labels=lables
        
    )

    ax[1].pie(
        data_2,
        startangle=90,
        colors = [color_1, color_2],
        autopct="%0.1f%%",
        wedgeprops={"edgecolor": "black"},
        textprops={"fontsize": size*1.5},
        labels=lables
    
    )

    ax[0].set_title(title_1, fontsize=size*2)
    ax[1].set_title(title_2, fontsize=size*2)


    plt.tight_layout()
    plt.subplots_adjust(wspace=1, hspace=1.0) 
    plt.show()




def create_confusion_matrix(pipeline, X_test, y_test, color, title):

    '''
    Creates a confusion matrix plot.
    '''

    from sklearn.metrics import ConfusionMatrixDisplay

    plt.rcParams.update(params)

    conf_matrix = ConfusionMatrixDisplay.from_estimator (pipeline, X_test, y_test, cmap=color)
    conf_matrix.ax_.set_xticks([0, 1])
    conf_matrix.ax_.set_xticklabels(["No", "Yes"])
    conf_matrix.ax_.set_yticks([0, 1])
    conf_matrix.ax_.set_yticklabels(["No", "Yes"])

    plt.title(title)
    plt.show()




def create_feature_importance(pipeline, classifier, preprocessor):

    '''
    Creates a DataFrame of feature importances.
    '''
    
    model = pipeline.named_steps[classifier]

    feature_names = pipeline.named_steps[preprocessor].get_feature_names_out()

    importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    print(feature_importance_df)



  
def create_classification_report(pipeline, X_test, y_test):
    
    '''
    Creates a classification report.
    ''' 

    from sklearn.metrics import classification_report
    
    y_predict = pipeline.predict(X_test)
    print(classification_report(y_test, y_predict))

    return y_predict


def create_AUC(pipeline, X_test, y_test):

    '''
    Creates an AUC score.
    '''

    from sklearn.metrics import roc_auc_score
    y_score = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_score)

    print(f'Area Under the Curve (AUC): {auc_score:.2f}')
    
    


def create_ROC(pipeline, X_test, y_test):
    
    '''  
    Creates a ROC curve.
    '''

    from sklearn.metrics import roc_curve, auc
    plt.rcParams.update(params)

    y_score = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()



