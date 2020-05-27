# Credit-Card-Fraud-ML
## Contributors
* Kyle Felkel
* Micheal Chan

## Description
This is a machine learning algorithm used to determine if credit card transactions are fraudulent. We used Random Forest and Naïve-Bayes Classifiers and applied each of them with 3 different selection methods, resulting in a total of 6 models. 

## Results
All of the models performed similarly, with Accuracy scores in the mid 90s. Details and visualizations can be found in the results folder.

##Data
The data set is a real world data set from transactions in Europe from September 2013. All features have been anonymized and normalized for privacy. The transaction data was given in a .csv format with a total of entries is 284, 807. The is a large data and can be found on kaggle [here](https://www.kaggle.com/mlg-ulb/creditcardfraud). 
Given the large data imbalannce, undersampling was used to improve model accuracy.

## Tools
Code was written in Python with the following libraries:
* Pandas to import the .csv and do mass operations
* matplotlib and seaborn for data visualizations
* sklearn for the ML part
* mlxtend for backward feature selection on the classifiers


