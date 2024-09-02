![Alt_Text](https://github.com/TuringCollegeSubmissions/knouri-ML.3/blob/main/Images/Titanic.PNG)
# Project Files
01 Overview.ipynb
02 EDA.ipynb
03 FeatureEngineering.ipynb
04 Imputations.ipynb
05 MachineLearning.ipynb
06 Conclusions.ipynb


# Introduction
In the year 2912 the interstellar spaceship Titanic has collided with a spacetime anomaly and some of the passengers were transported to an alternate dimension. In this analysis, I will use the records recovered from the spaceship’s computer system to predict which passengers were transported.

# Dataset
**Train file (spaceship_titanic_train.csv)** — Contains personal records of the passengers that would be used to build the machine learning model.

**Test file (spaceship_titanic_test.csv)** — Contains personal records for the remaining passengers, but not the target variable. It will be used to see how well our model performs on unseen data.

**Sample Submission File (sample_submission.csv)** — Contains the format to submit predictions.


# Goal
My goal is to get a score above 79 on the Kaggles's Spaceship Titanic competition.

# Project Folder
This folder contains 6 notebooks related to this project:
1. Overview
2. EDA
3. Feature Engineering
4. Missing Data Imputation
5. Machine Learning
6. Conclusions


# Technical Requirements
1. Exploratory data analysis
2. Pre-Processing of the data
3. Application of various machine learning models to predict which passengers were transported
4. Clear explanations of findings
5. Final conclusions
6. Suggestions on how the analysis can be improved


# Standards
> **Standard 1:** My standard for an acceptable accuracy score is 79%. <BR>
> **Standard 2:** My standard for colinnearity is a Pearson correlation coefficient of approximately 0.8. <BR> 

# Biases
The main bias is that some data is missing. Every feature except the PassengerId has over 2% missing data. I tried various imputing techniques to address this issue.

# Conclusions
>* **The Analysis of the Data:** I reviewed approximately 8,700 datapoint related to passnegers on the Spaceship Titanic. <br> 
>* **The Goal of the Project:** The goal of this project was to find a model that could predict if a passenger was transported to an alternative dimensions with an accuracy score over 79.<br>
>* **Missing Data:** There was over 2% missing data in every feature except PassengerId. However, I was able to impute some of this missing information using known data from other features. For example, from Last Name of the passenger it was possible to impute some of the Home Planet and some of the Destination that was missing.  <br>
>* **Croygenic Sleep:** Passengers who were in Cryogenic Sleep during the trip were less likely to be transported. Cryogenic Sleep turend out to be a very important feature. <br>
>* **Luxury Spending:** Passengers who spend money on luxury items like spa were more likely to be transported. Expenditure on luxury items turned out important in predicting Transportation.<br>
>* **Models:** I utilized the following models: Logistic Regression, K Nearest Neighbors, Random Forest, Extreme Gradient Boosting (XGB), Light Gradient Boosting Machine (LGBM), Categorical Boosting (CatBoost). The boosting models gave the best performance. I used GridSearch CV with the three boosting models to tune their hyperparameters. <br>
>* **Extremet Gradient Boosting (XGB):** Best Cross Validation Score: 0.81. Best Test Score: 0.78. Modeling Time: 22 minutes   <br>
>* **Light Gradient Boosting (LGBM):** Best Cross Validation Score: 0.81. Best Test Score: 0.79. Modeling Time: 2 minutes.  <br>
>* **Categorical Boosting (CatBoost):** Best Cross Validation Score: 0.81. Best Test Score: 0.78. Modeling Time: 71 minutes.<br>
>* **Final Model (LGBM):** I used LGBM as my final model. I obtained a score of 0.80032 on the Kaggle competition.<br>

# Suggestions for Improvement
>* **Domain Knowledge:** It is best if the data scientist has adequate domain knowledge on the topic of the analysis. I do not have any expertise in space travel or alternative dimensions. There may be parts of the data that I have overlooked that may have been important and I may have given importance to parts that may have had little significance. <br>
>* **More Complete Data:** As mentioned earlier, there was missing data in every feature. Less missing data could have improved the performance of the models.<br>  
>* **Pandas:** Continue to learn to utilize more optimized Pandas techniques and algorithms.<br>
>* **Seaborn and Matplotlib:** Continue to improve my knowledge of Seaborn and Matplotlib for creating visualizations. <br>
>* **Machine Learning:** Continue to improve on my capabilities with various models. <br>
>* **Python Code:** Continue to write better and more efficient Python code. <br>
>* **Clean Code:** Continue to adhere to the principles of writing clean code. <br>
>* **Readability and Efficiency:** Continue to improve my skills to find the delicate balance between readability and efficiency in coding.<br>