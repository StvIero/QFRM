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


# start PCA analysis here, kinda want to have it set up in a way so i can automate it more easily next time:
def PCA(df, ticker_list):
    # packages again:
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
        
    ticker_list = ['ASML.AS', 'SONY', 'AKZA.AS', 'BAYN.DE', 'TSN', 'NTDOY', 'SQNXF', 'AMD','CSGN.SW', 'MUFG']
    dfrets = df.iloc[1:,len(ticker_list)+1:len(df.columns)-1]
    
    # now just use PCA analysis on the dfrets to see some dependencies:
    pca = PCA(n_components='mle', svd_solver = 'full', )
    pca_fit = pca.fit(dfrets)
    
    loadings = pd.DataFrame(pca.components_)
    
    # get barplot of first couple components:
    # somehow col 0 and 1 are the exact same in loadings df, kind of weird?
    print(np.cumsum(pca.explained_variance_ratio_)) # 90% after the 6th one...
    
    # plot of cumsum ===================================
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Nr of principal component')
    plt.ylabel('% of explained variance')
    plt.show()
    
    
    # some barplots menn
    loadingsneg = loadings*-1

    fig,axs = plt.subplots(2,3)
    axs[0,0].bar(x= loadings.index.values, height=loadingsneg.iloc[:,1],label='pc1')
    axs[0,1].bar(x= loadings.index.values, height=loadingsneg.iloc[:,2],label='pc2')
    axs[0,2].bar(x= loadings.index.values, height=loadingsneg.iloc[:,3],label='pc3')
    axs[1,0].bar(x= loadings.index.values, height=loadingsneg.iloc[:,4],label='pc4')
    axs[1,1].bar(x= loadings.index.values, height=loadingsneg.iloc[:,5],label='pc5')
    axs[1,2].bar(x= loadings.index.values, height=loadingsneg.iloc[:,6],label='pc6') # overlap but plox
    plt.tight_layout()
    
    
    return loadings, dfrets, pca_fit


def FA(df, tickerlist):
    # packages again:
    from sklearn.decomposition import SparsePCA
    
    
    return 1








def Biv_copulas(df, tickerlist, copula_dist):
    # copula_dist can be:
    # [normal, student, clayton, frank, gumbel]
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    ticker_list = ['ASML.AS', 'SONY', 'AKZA.AS', 'BAYN.DE', 'TSN', 'NTDOY', 'SQNXF', 'AMD','CSGN.SW', 'MUFG']
    dfrets = df.iloc[1:,len(ticker_list)+1:len(df.columns)-1]
    
    
    #for i in [1,5,8]:
    for i in [1]:
        dfcop = dfrets.iloc[:,i:i+2]
        _,ndim = dfcop.shape
        plt.scatter(dfcop.iloc[:,0], dfcop.iloc[:,1])
        plt.title(str(dfcop.columns[0])+' and '+str(dfcop.columns[1]))

        if copula_dist == 'normal':
            from copulae import GaussianCopula
            g_cop = GaussianCopula(dim=ndim)
            g_cop.fit(dfcop)
            print('params are', g_cop.params)
            
        elif copula_dist == 'student':
            from copulae import StudentCopula
            t_cop = StudentCopula(dim=ndim)
            t_cop.fit(dfcop)
            print('params are', t_cop.params)
            
        elif copula_dist == 'clayton':
            from copulae import ClaytonCopula
            c_cop = ClaytonCopula(dim=ndim)
            c_cop.fit(dfcop)
            print('params are', c_cop.params)
            
        elif copula_dist == 'frank':
            from copulae import FrankCopula
            f_cop = FrankCopula(dim=ndim)
            f_cop.fit(dfcop)
            print('params are', f_cop.params)
            
        elif copula_dist == 'gumbel':
            from copulae import GumbelCopula
            g_cop = GumbelCopula(dim=ndim)
            g_cop.fit(dfcop)
            print('params are', g_cop.params)
            
    return dfcop

'''====================================== RUN HERE =================================='''

df = DataPuller_assignment3() # get data

# loadings, dfrets, pca = PCA(df, 0)


test_cop = Biv_copulas(df, 0, 'gumbel')







