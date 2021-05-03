# -*- coding: utf-8 -*-
"""
Pull some data for 10 stocks used for PCA FA and copula and extreme value theory later...

Created on Mon May  3 15:50:57 2021

@author: MauritsOever
"""

def DataPuller_assignment3():
    # specify packages just in case:
    from pandas_datareader import data as pdr
    from datetime import date
    import yfinance as yf
    yf.pdr_override()
    import pandas as pd
    # hardcode all the arguments bc one assignment anyways
    ticker_list = ['ASML.AS', 'SONY', 'AKZA.AS', 'BAYN.DE', 'TSN', 'NTDOY', 'SQNXF', 'AMD','CSGN.SW', 'MUFG']
    today = date.today()
    start_date = '2011-04-20'
    end_date = '2021-04-20'
    files = []
    
    def SaveData(df, filename):
        df.to_csv(r"C:\Users\gebruiker\Documents\GitHub\QFRM\Data3\\" +filename+".csv")
        
    
    def getData(ticker):
        print(ticker)
        data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
        dataname = ticker
        files.append(dataname)
        SaveData(data, dataname)
        
    for tik in ticker_list:
        getData(tik)
    
    
    df = pd.read_csv(r"C:\Users\gebruiker\Documents\GitHub\QFRM\Data3\\"+str(files[1])+".csv")
    df[str(files[0])] = df['Adj Close']
    # filter df on adjclose and date:
    df = df.iloc[:,list([0,-1])]
    
    for i in range(1, len(files)):
    #for i in range(1, 3):
        df1 = pd.read_csv(r"C:\Users\gebruiker\Documents\GitHub\QFRM\Data3\\"+str(files[i])+".csv")
        df1[str(files[i])] = df1['Adj Close']
        df1 = df1.iloc[:,list([0,-1])]
        
        # now join those df1s to df for master dataset to get 
        df = pd.merge(df, df1, how='left', on=['Date'])
        
    print(df.head(1))
    return df



df = DataPuller_assignment3()
df.to_csv(r"C:\Users\gebruiker\Documents\GitHub\QFRM\Data3\data_main.csv")

