# -*- coding: utf-8 -*-
"""
This file will do PCA analysis and such
some FA too

Created on Tue May  4 11:50:54 2021

@author: MauritsOever
"""

# packages and set directory
import os
os.chdir(r"C:\Users\gebruiker\Documents\GitHub\QFRM") # necessary to get dataloader function
from data_puller3000 import DataPuller_assignment3 # get dataloader function
import matplotlib as plt


# start PCA analysis here, kinda want to have it set up in a way so i can automate it more easily next time:
def PCA(df, ticker_list):
    # packages again:
    from sklearn.decomposition import PCA
    import pandas as pd
        
    ticker_list = ['ASML.AS', 'SONY', 'AKZA.AS', 'BAYN.DE', 'TSN', 'NTDOY', 'SQNXF', 'AMD','CSGN.SW', 'MUFG']
    dfrets = df.iloc[1:,len(ticker_list)+1:len(df.columns)-1]
    
    # now just use PCA analysis on the dfrets to see some dependencies:
    pca = PCA(n_components='mle', svd_solver = 'full', )
    pca_fit = pca.fit(dfrets)
    
    loadings = pd.DataFrame(pca.components_)
    
    # get barplot of first couple components:
    # somehow col 0 and 1 are the exact same in loadings df, kind of weird?
    print(pca.explained_variance_ratio_)
    
    
    return loadings, dfrets, pca_fit




'''====================================== RUN HERE =================================='''

df = DataPuller_assignment3() # get data



loadings, dfrets, pca = PCA(df, 0)


