# -*- coding: utf-8 -*-
"""
Pull some data for 10 stocks used for PCA FA and copula and extreme value theory later...

Created on Mon May  3 15:50:57 2021

@author: MauritsOever
"""

def DataPuller_assignment3():
    # specify packages just in case:
    import os
    import numpy as np
    import pandas as pd
    from pandas_datareader import data as pdr
    from datetime import date
    import yfinance as yf
    yf.pdr_override()
    
    # hardcode all the arguments bc one assignment anyways, next time i can automate more easily
    # main args
    ticker_list = ['ASML.AS', 'SONY', 'AKZA.AS', 'BAYN.DE', 'TSN', 'NTDOY', 'SQNXF', 'AMD','CSGN.SW', 'MUFG']
    path = r"C:\Users\gebruiker\Documents\GitHub\QFRM\Data3\\"
    
    # some more args
    start_date = '2011-04-20'
    end_date = '2021-04-20'
    files = []
    
    # check if function has run/downloaded stuff before:
    if 'data_main.csv' in os.listdir(r"C:\Users\gebruiker\Documents\GitHub\QFRM\Data3\\"):
        df = pd.read_csv(r"C:\Users\gebruiker\Documents\GitHub\QFRM\Data3\data_main.csv",index_col=0)
        
    else:
        def SaveData(df, filename):
            df.to_csv(path +filename+".csv")
            
        
        def getData(ticker):
            print(ticker)
            data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
            dataname = ticker
            files.append(dataname)
            SaveData(data, dataname)
            
        for tik in ticker_list:
            getData(tik)
        
        
        df = pd.read_csv(path+str(files[1])+".csv")
        df[str(files[0])] = df['Adj Close']
        # filter df on adjclose and date:
        df = df.iloc[:,list([0,-1])]
        
        for i in range(1, len(files)):
        #for i in range(1, 3):
            df1 = pd.read_csv(path+str(files[i])+".csv")
            df1[str(files[i])] = df1['Adj Close']
            df1 = df1.iloc[:,list([0,-1])]
            
            # now join those df1s to df for master dataset to get 
            df = pd.merge(df, df1, how='left', on=['Date'])
        
        # clean it up a bit, remove nans by ffill
        df = df.iloc[1:,:]
        df = df.ffill(axis=0)
    
        # get log returns for every ticker
        
        for tic in df.columns[1:]:
            df[tic+'_ret'] = np.log(df[tic]) - np.log(df[tic].shift(1))
            
        # get some portfolio returns, assume average weight...
        df['port_ret'] = df.iloc[:,len(ticker_list)+1:len(df.columns)+1].mean(axis=1)
        df.to_csv(path+'data_main.csv')
    
    dfrets = df.iloc[1:,len(ticker_list)+1:len(df.columns)-1]
    return df, dfrets



df, dfrets = DataPuller_assignment3()





