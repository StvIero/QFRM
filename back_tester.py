# -*- coding: utf-8 -*-
"""
This file will put out some backtesting stats based on a df which has the columns
VaR, ES, Loss. It wil calc violations and do backtesting based on those...

Created on Fri Apr 30 12:04:04 2021

@author: MauritsOever
"""

# packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import matplotlib.dates as mdates

# import data to test with, its the function that we're interested in:
dframe = pd.read_csv(r'C:\Users\gebruiker\Desktop\VU\Master\QFRM\var_es975CCCn.csv', index_col=0)
dframe['Date'] = pd.date_range(start='1/1/2012', periods=len(dframe), freq='D') # get pseudo dates to test with




# if index = date
def backtester_indate(df, alpha, g_label):
    # alpha needs to be put in as 0.01 or 0.025
    # g_label is the label you want your graphs to have, e.g. '97.5% VaR historical simulation'
    
    df['diff'] = df.iloc[:,0] - df.iloc[:,2] #difference column
    df['vio'] = np.full((len(df),1), 0) 
    df.loc[df['diff']<0,'vio'] = 1 # get violation dummy
    
    df['Year'] = pd.DatetimeIndex(df['Date']).year #change date column to just its years
    # so now column 0 = VaR, 1= ES, 2=Loss, 3=date/year, 4=diff, and 5=violations
    
    # starting with violation amount... test with binom statistical test
    for t in df['Year'].unique():
        subset = df.loc[df['Year'] == t]
        nr_vios = np.sum(subset['vio'])
        
        
        print(t,':')
        print()
        print('amount of violations is ', nr_vios, ', ratio is', nr_vios/len(subset['vio']))
        print('p-value is ', stats.binom_test(nr_vios, n=len(subset['vio']), p=alpha, alternative='greater'))
        
        # now we are gonna get actual shortfall
        print('realized shortfall is', np.mean(subset.loc[subset['diff']<0, '2']))
        print()
    
    
    
    # lets also do a plot of rets and VaR values...
    index = pd.to_datetime(df.iloc[1:-1, 3])
    print(type(index))
    
    #fig, ax = plt.subplot()
    plt.plot(index, df.iloc[1:-1, 0], label='VaR')
    plt.plot(index, df.iloc[1:-1, 1], label='ES')
    plt.plot(index, df.iloc[1:-1, 2], alpha=0.5, label='Return')
    plt.legend()
    
    #plt.xaxis.set_tick_params(reset=True)
    #plt.xaxis.set_major_locator(mdates.YearLocator(1))
    #plt.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    
    
    
    # now we will plot some graphs for visual testing
    loi = []
    for i in range(len(df)-1):
        if df.loc[i,'diff'] < 0:
            loi.append(i)
    
    indiffs = np.array(loi)[1:] - np.array(loi)[:-1]
    # now for the QQplot:
    sm.qqplot(indiffs, stats.expon,fit=True, line='45', label= g_label)
    
    return df


test = backtester_indate(dframe, 0.01, '97.5% VaR historical simulation')




# same thing but if date is a column