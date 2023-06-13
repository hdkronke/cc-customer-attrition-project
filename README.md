# Project 4: Predicting Credit Card Customer Attrition
#### Group 1: Alex Delacruz, Bennett Northcutt, Emily Neaville, Hays Kronke, and Stephen Mims
#### Dataset: https://zenodo.org/record/4322342#.Y8OsBdJBwUE
---
# Research question:
## Can we predict customer churn for the bank's credit card customers in order to reduce the rate of attrition?

# Summary:
Our group will use the dataset to build  machine learning models that can accurately predict bank customers who are at risk of attrition. This dataset contains customer information ranging from demographic (age, gender, education) to financial data (income bracket, card history, credit limit). Using these features, our aim is to build a model that will be able to be used by bankers, sales managers, branch managers, or other decision makers to help the bank reduce customer churn.

## Processes included in Jupyter Notebook:
1. Data Loading and EDA
2. Data preprocessing
3. Fitting models and making predictions
4. KNN, Logistic Regression, and Random Forest results
5. Adjusted weights and oversampled Logistic Regression models
6. Feature selection
7. Optimized models using feature importance

Libraries used: pandas, sqlite3, seaborn, matplotlib, numpy, scipy, scikit-learn

## Data loading and EDA
After reading in and cleaning the data, we were able to conduct some exploratory analysis. The main takeaway from this in regard to building our machine learning models was taking note of the imbalanced classes. There were many more instances of existing customers than there were of attrited customers, as shown in this visualization.
![attrition_bar](https://github.com/hdkronke/Project4/blob/main/Figures/attrition_bar.png)

## Data preprocessing
- Utilized scipy to remove outliers using z-scores
- Encoded both the target variables (Attrition_Flag) and categorial features
- Split testing and training data
- Scaled the data

## Model results
We built a KNN model, a random forest model, and a logisitic regression model. The results of the models can be seen below:
![KNN](https://github.com/hdkronke/Project4/blob/main/Figures/KNN.png)
![random forest](https://github.com/hdkronke/Project4/blob/main/Figures/RandomForest.png)
![Logistic Regression](https://github.com/hdkronke/Project4/blob/main/Figures/LogisticRegression.png)

## Feature Importance
Creating the random forest model allows us to identify the most important features of the model
After visualizing the important features, we create a new dataframe dropping some features and retrained our models for performance improved.
![feature importance](https://github.com/hdkronke/Project4/blob/main/Figures/feature_importances.png)

## Final model
After retraining theKNearestNeighbors, the logistic regression, and the random forest models with the selected features, the random forest model still maintained the best performance. Using feature selection, we were able to slightly increase the accuracy by 1% and the recall of the model, which went from 84% to 87%. There was a slight decrease in the precision of the model, but in the end that was a hit we were willing to take for the improved performance. The confusion matrix of the optimized model of choice can be seen below.
![optimized random forest model](https://github.com/hdkronke/Project4/blob/main/Figures/RF_optimized.png)
