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
    
    
    return loadings, pca_fit



def FA(dfrets, tickerlist):
    # packages again:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import SparsePCA
    from sklearn.decomposition import FactorAnalysis
    
    
    fa = FactorAnalysis(n_components=len(dfrets.columns), random_state=0)
    fa_fit = fa.fit(dfrets)
    loadings = pd.DataFrame(fa_fit.components_)
    loadingsneg = loadings*-1

    fig,axs = plt.subplots(2,3)
    axs[0,0].bar(x= loadings.index.values, height=loadingsneg.iloc[:,1],label='pc1')
    axs[0,1].bar(x= loadings.index.values, height=loadingsneg.iloc[:,2],label='pc2')
    axs[0,2].bar(x= loadings.index.values, height=loadingsneg.iloc[:,3],label='pc3')
    axs[1,0].bar(x= loadings.index.values, height=loadingsneg.iloc[:,4],label='pc4')
    axs[1,1].bar(x= loadings.index.values, height=loadingsneg.iloc[:,5],label='pc5')
    axs[1,2].bar(x= loadings.index.values, height=loadingsneg.iloc[:,6],label='pc6') # overlap but plox
    plt.tight_layout()
    
    #print(np.cumsum(fa_fit.explained_variance_ratio))
    
    return loadings



def Biv_copulas(dfrets, tickerlist, copula_dist):
    # copula_dist can be:
    # [normal, student, clayton, frank, gumbel]
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    ticker_list = ['ASML.AS', 'SONY', 'AKZA.AS', 'BAYN.DE', 'TSN', 'NTDOY', 'SQNXF', 'AMD','CSGN.SW', 'MUFG']    
    
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

df,dfrets = DataPuller_assignment3() # get data

test = FA(dfrets, 1)

loadings, pca_fit = PCA(dfrets, 1)

































































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







