# -*- coding: utf-8 -*-
"""
Credit Score logistic regression application...
Pretty straight forward method of predicting default prob -->
Load in a data set from Kaggle:
https://www.kaggle.com/c/GiveMeSomeCredit/data?select=cs-training.csv

Created on Mon May 24 12:19:11 2021

@author: MauritsOever
"""

# packages
import pandas as pd



"""
Data part, load in and clean NA's
"""
# load in data:
train = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\QFRM\Assignment 4+5\cs-training.csv')
test = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\QFRM\Assignment 4+5\cs-test.csv')
# SeriousDLQin2yrs is target variable
# dont have true y for test set so might just ignore...

# check for NaNs
train.isna().sum() # monthly income and number of dependents have NaNs
test.isna().sum() # SeriousDlqin2yrs, MonthlyIncome, NumberOfDependents
# not too bad

# bad practice in this case but fill with means for now:
train = train.fillna(train.mean())
test = test.fillna(test.mean())

"""
Feature engineering part:
    - missing_income_bool
    - 

"""
columns = ['SeriousDlqin2yrs',
       'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']

real_ytrain = train[columns[0]]
xtrain = train[columns[1:]]
xtest = test[columns[1:]]



"""
Fit logistic regression here:
"""
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(n_jobs=-1, max_iter=1000)
logreg = logreg.fit(xtrain, real_ytrain)
# okay we fitted, lets predict
pred_ytrain = logreg.predict_proba(xtrain) #used to compare to real defaults...


"""
Some code here for cross validation or k-fold stuff or whatever
"""



