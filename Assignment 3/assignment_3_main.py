# -*- coding: utf-8 -*-
"""
This file will do PCA analysis and such
some FA too

Created on Tue May  4 11:50:54 2021

@author: MauritsOever
"""

# packages and set directory
import copulas
import os
os.chdir(r"C:\Users\gebruiker\Documents\GitHub\QFRM\Assignment 3") # necessary to get dataloader function
from data_puller3000 import DataPuller_assignment3 # get dataloader function


def sumstats(dfrets):
    import numpy as np
    import scipy.stats as stats
    for i in dfrets.columns:
        print(str(i)+': ', stats.kurtosis(dfrets[i]))
    
    return dfrets.describe()

# start PCA analysis here, kinda want to have it set up in a way so i can automate it more easily next time:
def PCA(df):
    # packages again:
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
        
    ticker_list = ['ASML.AS', 'SONY', 'AKZA.AS', 'BAYN.DE', 'TSN', 'NTDOY', 'SQNXF', 'AMD','CSGN.SW', 'MUFG']    
    # now just use PCA analysis on the dfrets to see some dependencies:
    pca = PCA(n_components=10, svd_solver = 'full') #n_components='mle',
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
    loadingsneg = loadings#*-1
    loadingsnegT = loadingsneg.transpose()
    lables = dfrets.columns

    fig,axs = plt.subplots(2,3,figsize=(10,8))
    axs[0,0].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,0],label='pc1')
    axs[0,0].set_title('PC1')
    axs[0,1].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,1],label='pc2')
    axs[0,1].set_title('PC2')
    axs[0,2].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,2],label='pc3')
    axs[0,2].set_title('PC3')
    axs[1,0].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,3],label='pc4')
    axs[1,0].set_title('PC4')
    axs[1,1].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,4],label='pc5')
    axs[1,1].set_title('PC5')
    axs[1,2].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,5],label='pc6') # overlap but plox
    axs[1,2].set_title('PC6')
    
    plt.sca(axs[0, 0])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    plt.sca(axs[0, 1])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    plt.sca(axs[0, 2])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    plt.sca(axs[1, 0])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    plt.sca(axs[1, 1])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    plt.sca(axs[1, 2])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    #plt.xticks(loadingsnegT.index.values, labels, rotation=90)
    plt.tight_layout()
    plt.show()
    
    
    print(loadingsnegT.to_latex())
    
    return loadingsnegT, pca_fit



def FA(dfrets):
    # packages again:
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import FactorAnalysis
    
    # fit
    fa = FactorAnalysis(n_components=7,rotation='varimax')
    fa_fit = fa.fit(dfrets)
    loadings = pd.DataFrame(fa_fit.components_)
    loadings_t = loadings.transpose()
    loadingsnegT = loadings_t#*-1
    
    # plot
    lables = dfrets.columns

    fig,axs = plt.subplots(2,3,figsize=(10,8))
    axs[0,0].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,0],label='F1')
    axs[0,0].set_title('F1')
    axs[0,1].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,1],label='F2')
    axs[0,1].set_title('F2')
    axs[0,2].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,2],label='F3')
    axs[0,2].set_title('F3')
    axs[1,0].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,3],label='F4')
    axs[1,0].set_title('F4')
    axs[1,1].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,4],label='F5')
    axs[1,1].set_title('F5')
    axs[1,2].bar(x= loadingsnegT.index.values, height=loadingsnegT.iloc[:,5],label='F6') # overlap but plox
    axs[1,2].set_title('F6')
    
    plt.sca(axs[0, 0])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    plt.sca(axs[0, 1])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    plt.sca(axs[0, 2])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    plt.sca(axs[1, 0])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    plt.sca(axs[1, 1])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    plt.sca(axs[1, 2])
    plt.xticks(loadingsnegT.index.values, lables, rotation=90)
    #plt.xticks(loadingsnegT.index.values, labels, rotation=90)
    plt.tight_layout()
    plt.show()
    
    print(loadingsnegT.to_latex())
    
    return loadings_t



def Biv_copulas_bad(dfrets, copula_dist):
    # copula_dist can be:
    # [normal, student, clayton, frank, gumbel]
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    ticker_list = ['ASML.AS', 'SONY', 'AKZA.AS', 'BAYN.DE', 'TSN', 'NTDOY', 'SQNXF', 'AMD','CSGN.SW', 'MUFG']    
    pair = [['CSGN.SW_ret', 'BAYN.DE_ret'], ['TSN_ret', 'BAYN.DE_ret'], ['AMD_ret', 'ASML.AS_ret'], ['SONY_ret','SQNXF_ret']]
    
    #for i in [1,5,8]:
    for i in [0]:
        dfcop = dfrets[pair[3]]
        _,ndim = dfcop.shape
        plt.scatter(dfcop.iloc[:,0], dfcop.iloc[:,1])
        plt.title(str(dfcop.columns[0])+' and '+str(dfcop.columns[1]))
        plt.show()

        if copula_dist == 'normal':
            from copulae.elliptical import GaussianCopula
            cop = GaussianCopula(dim=ndim)
            pobsdata = GaussianCopula.pobs(dfcop)
            cop.fit(dfcop)
            plt.scatter(pobsdata.iloc[:,0], pobsdata.iloc[:,1],alpha=0.5)
            plt.title(str(dfcop.columns[0])+' and '+str(dfcop.columns[1]))
            plt.show()
            
        elif copula_dist == 'student':
            from copulae.elliptical import StudentCopula
            cop = StudentCopula(dim=ndim)
            pobsdata = StudentCopula.pobs(dfcop)
            cop.fit(dfcop)
            plt.scatter(pobsdata.iloc[:,0], pobsdata.iloc[:,1],alpha=0.5)
            plt.title(str(dfcop.columns[0])+' and '+str(dfcop.columns[1]))
            plt.show()
            
        elif copula_dist == 'clayton':
            from copulae.archimedean import ClaytonCopula
            cop = ClaytonCopula(dim=ndim)
            pobsdata = ClaytonCopula.pobs(dfcop)
            cop.fit(dfcop)
            plt.scatter(pobsdata.iloc[:,0], pobsdata.iloc[:,1],alpha=0.5)
            plt.title(str(dfcop.columns[0])+' and '+str(dfcop.columns[1]))
            plt.show()
            
        elif copula_dist == 'frank':
            from copulae.archimedean import FrankCopula
            cop = FrankCopula(dim=ndim)
            pobsdata = FrankCopula.pobs(dfcop)
            cop.fit(dfcop)
            plt.scatter(pobsdata.iloc[:,0], pobsdata.iloc[:,1],alpha=0.5)
            plt.title(str(dfcop.columns[0])+' and '+str(dfcop.columns[1]))
            plt.show()
            
        elif copula_dist == 'gumbel':
            from copulae.archimedean import GumbelCopula
            cop = GumbelCopula(dim=ndim)
            pobsdata = GumbelCopula.pobs(dfcop)
            cop.fit(dfcop)
            plt.scatter(pobsdata.iloc[:,0], pobsdata.iloc[:,1],alpha=0.5)
            plt.title(str(dfcop.columns[0])+' and '+str(dfcop.columns[1]))
            plt.show()
        
    print('params are', cop.params)
    print('log lik =', cop._fit_smry.log_lik)
    
    return cop



def Biv_copulas_good(dfrets):
    import copulas
    pairs = [['CSGN.SW_ret', 'BAYN.DE_ret'], ['',''],['','']]
    
    
    
    
def EVT(dfrets):
    
    return 'This function is not complete yet...'
'''====================================== RUN HERE =================================='''

df,dfrets = DataPuller_assignment3() # get data
dfretsneg = dfrets*-1

#loadings = FA(dfretsneg)

#dfrets_flip = dfrets.iloc[:, ::-1]

test = Biv_copulas_bad(dfretsneg, 'gumbel')
# normal, student, clayton, frank, gumbel

# sumobject = sumstats(dfrets)

import numpy as np
from scipy import stats

np.mean(dfrets['SQNXF_ret'])
np.std(dfrets['SQNXF_ret'])

stats.t.fit(dfretsneg['SQNXF_ret'])

stats.t.ppf(0.025, 3.5, 0.00055, 0.0241)


























































# =============================================================================
# loadings, dfrets, pca = PCA(df, 0)
# 
# from copulalib.copulalib import Copula
# from scipy.stats import norm
# import matplotlib.pyplot as plt
# frank = Copula(dfrets.iloc[0:250,1],dfrets.iloc[0:250,2],family='gumbel')     
# uf,vf = frank.generate_uv(1000)    
#     
# plt.scatter(uf,vf,marker='.',color='blue')     
# plt.ylim(0,1)     
# plt.xlim(0,1)     
# plt.title('Frank copula')
# =============================================================================




# ==========================================copula lib stuff
#     # Random simulated data
# import numpy as np
# #x = np.random.normal(size=250)
# #y = 2.5*x + np.random.normal(size=250)
# 
# x = dfrets.iloc[0:250,7]
# y = dfrets.iloc[0:250,1]
# fig = plt.figure()
# 
# 
# 
# # Frank
# frank = Copula(x,y,family='frank')
# uf,vf = frank.generate_uv(1000)
# fig.add_subplot(2,2,1)
# plt.scatter(uf,vf,marker='.',color='blue')
# plt.ylim(0,1)
# plt.xlim(0,1)
# plt.title('Frank copula')
# 
# # Clayton
# clayton = Copula(x,y,family='clayton')
# uc,vc = clayton.generate_uv(1000)
# fig.add_subplot(2,2,2)
# plt.scatter(uc,vc,marker='.',color='red')
# plt.ylim(0,1)
# plt.xlim(0,1)
# plt.title('Clayton copula')
# 
# # Gumbel
# gumbel = Copula(x,y,family='gumbel')
# ug,vg = gumbel.generate_uv(1000)
# fig.add_subplot(2,2,3)
# plt.scatter(ug,vg,marker='.',color='green')
# plt.ylim(0,1)
# plt.xlim(0,1)
# plt.title('Gumbel copula')
# 
# plt.show()
# =============================================================================


# ===================================================================copulas package
# from copulas.datasets import sample_trivariate_xyz
# from copulas.multivariate import GaussianMultivariate
# from copulas.visualization import compare_3d
# import copulas

# # Load a dataset with 3 columns that are not independent
# real_data = sample_trivariate_xyz()

# # Fit a gaussian copula to the data
# copula = GaussianMultivariate()
# copula.fit(real_data)

# # Sample synthetic data
# synthetic_data = copula.sample(len(real_data))

# # Plot the real and the synthetic data to compare
# copulas.visualization.compare_3d(real_data, synthetic_data)


# # =============================================================================


# test_cop = Biv_copulas(df, 0, 'clayton')







