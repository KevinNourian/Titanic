![Alt_Text](https://github.com/KevinNourian/Stroke/blob/main/Image/Stroke.PNG)
# Project Files
01 Overview.ipynb
02 EDA.ipynb
03 FeatureEngineering.ipynb
04 Imputations.ipynb
05 MachineLearning.ipynb
06 Conclusions.ipynb


# Introduction
In the year 2912 the interstellar Spaceship Titanic has collided with a spacetime anomaly. Some of the passengers were transported to an alternate dimension. In this analysis, I will use the records recovered from the spaceship’s computer system to predict which passengers were affected.

# Dataset
**Train file (spaceship_titanic_train.csv)** — Contains personal records of the passengers that would be used to build the machine learning model.

**Test file (spaceship_titanic_test.csv)** — Contains personal records for the remaining passengers, but not the target variable. It will be used to see how well our model performs on unseen data.

**Sample Submission File (sample_submission.csv)** — Contains the format to submit predictions.


# Goal
My goal is to get a high score on the Spaceship Titanic competition.

# Notebooks Folder
This folder contains the various versions of this project. 

# Technical Requirements
1. Exploratory data analysis
2. Pre-Processing of the data
3. Application of various machine learning models to predict which passengers were transported
4. Clear explanations of findings
5. Final conclusions
6. Suggestions on how the analysis can be improved


# Standards
> **Standard 1:** My standard for an acceptable accuracy score is approximately 80%. <BR>
> **Standard 2:** My standard for colinnearity is a Pearson correlation coefficient of approximately 0.8. <BR> 

# Biases
The main bias is that approximately 25% of the data is missing. Every feature except the PassengerId feature has over 2% missing data. 

# Conclusions
>* **The Analysis of the Data:** I reviewed over 5,000 datapoint related to patients with stroke. <br> 
>* **The Goal of the Project:** The goal of this project was to find a model that could predict if a patient is likely to have a stroke with a recall score of 0.75 or higher.<br>
>* **Models:** I utilized numerous models and numerous ways to hyperparameter tuning.  I chose two simple models, Logistic Regressin and Support Vector Machines. I chose two boosting classifiers, LGBM and XGB and two ensemble models: Random Forest Classifier and Gradient Boosting Classifier.<br>
>* **Encoding:** For categorical data, I tried both Label Encoding and One-Hot Encoding. I did not see significant differences. I chose Label Encoding since the resulting table was more readable.  <br>
>* **Imputing Missing Data:** For imputing missing data, I tried mean, median, zero and random imputers. I saw no signinficant difference between them in the predictions of my models. I chose Randmo Imputer, since it I thought with such little information about the participants, it makes no sense to make any judgments about their features.  <br>
>* **Feature Engineering and Hyperparameter Testing:** I tried feature engineering and hyperparameter testing with techniques such as Backward Elimination, SHAP and OPTUNA. Some, I included in this report and some I didn't for sake of brevity. None of the measures I utilized improved resutls significanlty.<br> 
>* **Support Vector Machines:** For a simple model and using only default hyperparameters, SVC was able to get better or similar results than any other model, including the more complex ones.<br>  
>* **Boosting Models:** Of the boosting models that I utilized, none of them performed better than SVC.
>* **Ensemble Models:** Of the ensemble models that I utilized, none of them performed better than SVC.
>* **Recommendation:** I am not able to make any medical recommendations based on this data. However, some obvious elements that I was sure would contribute to the risk of stroke such as smoking or BMI, turned out to be a very poor predictors. <br> 

# Suggestions for Improvement
>* **Domain Knowledge:** It is best if the data scientist has adequate domain knowledge on the topic of the analysis. I do not have any expertise in the medical field. There may be parts of the data that I have overlooked that may have been important and I may have given importance to parts that may have had little significance. <br>
>* **More Detailed Data:** The data provide only general information on patients. This information is not adequate to predict a disease as complex as stroke. Information such as family history, genetic markers, blood trace element markers and more are missing in this data. More detailed information could have helped make better predictions. <br>  
>* **Balance:** The data is heavely imbalanced. Of the more than 5,000 datapoints, only about 300 are related to stroke patients. This, in addition to inadequcy of the data as mentioned above adds to the complexity predictions.  <br>  
>* **Visualizations:** If I had more time, I would improve on the bar graphs to emphasize certain data by using specific colors.  <br>  
>* **Functions:** I modularized most of the code in this notebook but not all due to limitation of time.  <br>  
>* **Statistics:** Continue to improve my statistical knowledge to create better analyses.<br>
>* **Pandas:** Continue to learn to utilize more optimized Pandas techniques and algorithms.<br>
>* **Seaborn and Matplotlib:** Continue to improve my knowledge of Seaborn and Matplotlib for creating visualizations. <br>
>* **Python Code:** Continue to write better and more efficient Python code. <br>
>* **Clean Code:** Continue to adhere to the principles of writing clean code. <br>
>* **Readability and Efficiency:** Continue to improve my skills to find the delicate balance between readability and efficiency in coding.<br>
>* **Functions File:** For my next project, I will create a file with my functions, separate from the notebook file, to keep the notebook as a more reasonable length.<br>