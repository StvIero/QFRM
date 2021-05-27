# -*- coding: utf-8 -*-
"""
Credit Score logistic regression application...
Pretty straight forward method of predicting default prob -->
Load in a data set from Kaggle:
https://www.kaggle.com/c/GiveMeSomeCredit/data?select=cs-training.csv

Created on Mon May 24 12:19:11 2021

@author: MauritsOever

To Do:
    - get information values (IVs) of all variables boiiii
    - get feature engineering, like get new feature, fill nans, get smote, then divide into xtrain, xtrain_smote, xtrain_smote_newfeatures
    - get output for logreg for all these guys and get metrics
    - check SHAP plots and staff, and then select most important ones xdd (take IV into account)
"""

# packages
import pandas as pd
import numpy as np
from icecream import ic



"""
Data part, load in and clean NA's
"""
# load in data:
train = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\QFRM\Assignment 4+5\cs-training.csv')

# check for NaNs, plot nan amounts
# import matplotlib.pyplot as plt
# nanamounts = train.isna().sum().values[1:] # monthly income and number of dependents have NaNs
# plt.bar(train.columns[1:], nanamounts)
# plt.xticks(rotation = 90)
# #plt.tight_layout()
# plt.show()


"""
Feature engineering part:
    - missing_income_bool
    - mssing_number_dependents_bool
    - 
    
    basically make a function that returns:
        - xtrain without extra features but with nans filled, and ytrain
        - xtrain with extra features and with nans filled, and ytrain
        - xtrain with extra features but drop the bad ones by SHAP...

"""

def feature_engineering(train):
    columns = ['SeriousDlqin2yrs',
           'RevolvingUtilizationOfUnsecuredLines', 'age',
           'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
           'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
           'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
           'NumberOfDependents']
    
    ytrain = train[columns[0]]
    xtrain = train[columns[1:]]
    #xtest = test[columns[1:]]
    
    # impute NA's
    train['MonthlyIncome'] = train.MonthlyIncome.fillna(0.0)
    train['NumberOfDependents'] = train.NumberOfDependents.fillna(train.NumberOfDependents.quantile(0.995))
    
    # SMOTE https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    # should make this a function...
    from imblearn.over_sampling import SMOTE
    oversample = SMOTE()
    xtrain_smote, ytrain_smote = oversample.fit_resample(xtrain, ytrain)
    # still need plots before and after smote
    
    # continue after smote with new feature engineering

    return ytrain, xtrain, ytrain_smote, xtrain_smote, ytrain_smote_newfeatures, xtrain_smote_newfeatures, xtrain_selected_features




"""
Fit logistic regression here:
    - could use full fit for interpretation
"""

def logreg_fit(xtrain, ytrain):
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(n_jobs=-1, max_iter=1000)
    logreg = logreg.fit(xtrain, ytrain)
    # okay we fitted, lets predict
    #pred_proba_ytrain = logreg.predict_proba(xtrain) #used to compare to real defaults...
    return logreg





"""
Some metrics
"""
def metrics(fitobject, xtrain, ytrain):
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_roc_curve
    from sklearn.metrics import confusion_matrix
    pred_ytrain = fitobject.predict(xtrain)
    plot_roc_curve(fitobject, xtrain, ytrain)
    plt.show()
    confusion = confusion_matrix(ytrain, pred_ytrain) 
    print('confusion matrix: \n', confusion)
    print('')
    
    tp = confusion[0,0]
    tn = confusion[1,1]
    fp = confusion[0,1]
    fn = confusion[1,0]

    print('recall = ', tp/(tp+fn))
    print('precision = ', tp/(tp+fp))
    print('accuracy = ', (tp+tn)/np.sum(confusion)) # very high? is that good?, probs need to balance with SMOTE
    
    


"""
cross-validation, wanna do 3-fold
"""

def crossvalidate(xtrain, ytrain, method):
    """
    3-fold cross-validation for the logistic regression/ML method

    Parameters
    ----------
    xtrain : all the independent variables, in a pandas.
    ytrain : the dependent variable to be predicted.
    method : 'logreg' or 'ML'

    Returns
    -------
    Not much, just prints comparisons between different folds
    and out of sample performances...

    """
    third = int(len(ytrain)/3)
    # create folds:
    x_in1 = xtrain.iloc[0:2*third,:]
    y_in1 = ytrain[0:2*third]
    x_out1 = xtrain.iloc[2*third:,:]
    y_out1 = ytrain[2*third:]
    
    x_in2 = xtrain.drop(xtrain.index[third:2*third])
    y_in2 = ytrain.drop(ytrain.index[third:2*third])
    x_out2 = xtrain.iloc[third:2*third,:]
    y_out2 = ytrain[third:2*third]
    
    x_in3 = xtrain.iloc[third:,:]
    y_in3 = ytrain[third:]
    x_out3 = xtrain.iloc[0:third,:]
    y_out3 = ytrain[0:third]
    
    
    if method == 'logreg':
        logreg1 = logreg_fit(x_in1, y_in1)
        logreg2 = logreg_fit(x_in2, y_in2)
        logreg3 = logreg_fit(x_in3, y_in3)
        
        print('Model 1: ')
        print('constant: ', logreg1.intercept_)
        print('params: ', logreg1.coef_)
        print('')
        print('Model 2: ')
        print('constant: ', logreg2.intercept_)
        print('params: ', logreg2.coef_)
        print('')
        print('Model 3: ')
        print('constant: ', logreg3.intercept_)
        print('params: ', logreg3.coef_)
        print('')
        
        print('model 1 metrics: ')
        metrics(logreg1, x_out1, y_out1)
        print('')
        print('model 2 metrics: ')
        metrics(logreg2, x_out2, y_out2)
        print('')
        print('model 3 metrics: ')
        metrics(logreg3, x_out3, y_out3)
        
        return logreg1, logreg2, logreg3


# =============================================================================
# Structure results like this:
#     - get normal fit without feature selection, no smote
#     - include smote
#     - include feature engineering
#     - include feature selection to avoid overfitting
# =============================================================================

# alldatasets = feature_engineering(train)
# then crossvalidate(in order)










"""
ML PART

Light GBM - check out
could also do
"""
# structure ML in the same way as 










