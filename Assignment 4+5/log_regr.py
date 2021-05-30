# -*- coding: utf-8 -*-
"""
Credit Score logistic regression application...
Pretty straight forward method of predicting default prob -->
Load in a data set from Kaggle:
https://www.kaggle.com/c/GiveMeSomeCredit/data?select=cs-training.csv

Created on Mon May 24 12:19:11 2021

@author: MauritsOever

To Do:
    - implement iv func
    - maybe more feature engineering
    - get output for logreg for all these guys and get metrics, and residual analysis
    - check SHAP plots and staff, and then select most important ones xdd (take IV into account)
"""

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
    
    xtrain_feat = train[columns[1:]]
    xtrain_feat['MissingIncomeBool'] = np.zeros(len(xtrain_feat))
    xtrain_feat['MissingDependentsBool'] = np.zeros(len(xtrain_feat))
    xtrain_feat.iloc[:,]
    
    xtrain_feat['MissingIncomeBool'][xtrain_feat.MonthlyIncome.isna()] = 1
    xtrain_feat['MissingDependentsBool'][xtrain_feat.NumberOfDependents.isna()] = 1
    
    
    # impute NA's
    xtrain['MonthlyIncome'] = xtrain.MonthlyIncome.fillna(0.0)
    xtrain['NumberOfDependents'] = xtrain.NumberOfDependents.fillna(train.NumberOfDependents.quantile(0.995))
    
    xtrain_feat['MonthlyIncome'] = xtrain.MonthlyIncome.fillna(0.0)
    xtrain_feat['NumberOfDependents'] = xtrain.NumberOfDependents.fillna(train.NumberOfDependents.quantile(0.995))
    
    # SMOTE https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    # should make this a function...
    from imblearn.over_sampling import SMOTE
    oversample = SMOTE()
    xtrain_smote, ytrain_smote = oversample.fit_resample(xtrain, ytrain)
    xtrain_feat_smote, ytrain_feat_smote = oversample.fit_resample(xtrain_feat, ytrain)
    
    # still need plots before and after smote
    
    # continue after smote with new feature engineering
    
    return ytrain, xtrain, ytrain_smote, xtrain_smote, ytrain_feat_smote, xtrain_feat_smote


"""
Fit logistic regression here:
"""

def logreg_fit(xtrain, ytrain):
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(n_jobs=-1, max_iter=1000)
    logreg = logreg.fit(xtrain, ytrain)
    # okay we fitted, lets predict
    #pred_proba_ytrain = logreg.predict_proba(xtrain) #used to compare to real defaults...
    return logreg


"""
Some metrics, To do: ROC AUC and F1
"""
def metrics(fitobject, xtrain, ytrain):
    # per 3-fold
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    pred_proba_ytrain = fitobject.predict_proba(xtrain)[:,1]

    pred_ytrain = fitobject.predict(xtrain)
    confusion = confusion_matrix(ytrain, pred_ytrain) 
    print('confusion matrix: \n', confusion)
    print('')
    
    tp = confusion[0,0]
    tn = confusion[1,1]
    fp = confusion[0,1]
    fn = confusion[1,0]
    
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)

    print('recall = ', recall)
    print('precision = ', precision)
    print('accuracy = ', (tp+tn)/np.sum(confusion)) # very high? is that good?, probs need to balance with SMOTE
    print('F1 = ', ((recall*precision)/(recall+precision))*2)
    print('ROC AUC = ', roc_auc_score(ytrain, pred_proba_ytrain))
    
    
"""
cross-validation, done
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
    permutations = np.random.permutation(len(xtrain))
    xtrain = xtrain.iloc[permutations,:]
    ytrain = ytrain[permutations]
    
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
        
        # print('Model 1: ')
        # print('constant: ', logreg1.intercept_)
        # print('params: ', logreg1.coef_)
        # print('')
        # print('Model 2: ')
        # print('constant: ', logreg2.intercept_)
        # print('params: ', logreg2.coef_)
        # print('')
        # print('Model 3: ')
        # print('constant: ', logreg3.intercept_)
        # print('params: ', logreg3.coef_)
        # print('')
        
        print('model 1 metrics: ')
        metrics(logreg1, x_out1, y_out1)
        print('')
        print('model 2 metrics: ')
        metrics(logreg2, x_out2, y_out2)
        print('')
        print('model 3 metrics: ')
        metrics(logreg3, x_out3, y_out3)
        
        return logreg1, logreg2, logreg3





def fit_and_residual_analysis(xtrain, ytrain):
    # per full set
    import shap
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_roc_curve
    from statsmodels.discrete.discrete_model import Logit
    from statsmodels.tools import add_constant
    logreg = logreg_fit(xtrain, ytrain)
    print('intercept: ', logreg.intercept_)
    print('params: ', logreg.coef_)
    
    plot_roc_curve(logreg, xtrain, ytrain)
    plt.show()
    
    
    # for logreg.summary(), we need :
        # - statsmodels.discrete.discrete_model.Logit
    model = Logit(ytrain, add_constant(xtrain)).fit(disp=0)
    print(model.summary().as_latex())
        # - result = model.fit(method='newton')
    
    # residual analysis goes here
    deviances = ytrain - logreg.predict_proba(xtrain)[:,1]
    fig, axs = plt.subplots(1,3, figsize=(12,5))
    
    axs[0].scatter(xtrain['age'],deviances)
    axs[0].set_title('Age')
    axs[1].scatter(xtrain['RevolvingUtilizationOfUnsecuredLines'],deviances)
    axs[1].set_title('Revolving Utilization Of Unsecured Lines')
    axs[2].scatter(xtrain['NumberOfTimes90DaysLate'],deviances)
    axs[2].set_title('Number Of Times 90 Days Late')
    plt.tight_layout()
    plt.show()
    
    
    print('')
    print('Feature selection: ')
    
    shapdata = shap.sample(xtrain, 80)
    explainer = shap.explainers.Exact(logreg.predict_proba, shapdata)
    shap_values = explainer(xtrain[:100])
    shap_values = shap_values[...,1]
    
    shap.plots.beeswarm(shap_values)
    shap.plots.waterfall(shap_values[2,:])
    
    
    return 
    
  
def feature_selection(xtrain_feat_smote, ytrain_feat_smote):    
    # get IV's and 
    from IV_code import data_vars
    
    iv = data_vars(xtrain_feat_smote, ytrain_feat_smote)
    print(iv[1].to_latex())
    new_columns = iv[1]['VAR_NAME'][-5:].values
    
    # just put manually based on SHAP and IV...
    # new_columns = ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTimes90DaysLate',
    #                'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse',
    #                'age']
    
    xtrain_selectedfeat_smote = xtrain_feat_smote[new_columns]
    ytrain_selectedfeat_smote = ytrain_feat_smote
    
    return xtrain_selectedfeat_smote, ytrain_selectedfeat_smote
    
  
# =============================================================================
# Structure results like this:
#     - get normal fit without feature selection, no smote
#     - include smote
#     - include feature engineering
#     - include feature selection to avoid overfitting
# =============================================================================





""" 
Thing u actually run
"""

# packages
import pandas as pd
import numpy as np
from icecream import ic

# load in and make data sets:
train = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\QFRM\Assignment 4+5\cs-training.csv')
ytrain, xtrain, ytrain_smote, xtrain_smote, ytrain_feat_smote, xtrain_feat_smote = feature_engineering(train) 
# ytrain, xtrain, _, _, _, _ = feature_engineering(train)
xtrain_selectedfeat_smote, ytrain_selectedfeat_smote = feature_selection(xtrain_feat_smote, ytrain_feat_smote)


#fit_and_residual_analysis(xtrain, ytrain)
#crossvalidate(xtrain, ytrain, 'logreg')

fit_and_residual_analysis(xtrain_smote, ytrain_smote)
crossvalidate(xtrain_smote, ytrain_smote, 'logreg')

fit_and_residual_analysis(xtrain_feat_smote, ytrain_feat_smote)
crossvalidate(xtrain_feat_smote, ytrain_feat_smote, 'logreg')


crossvalidate(xtrain_selectedfeat_smote, ytrain_selectedfeat_smote, 'logreg')
fit_and_residual_analysis(xtrain_selectedfeat_smote, ytrain_selectedfeat_smote)






"""
ML PART

Light GBM - check out
could also do
"""
# structure ML in the same way as 










