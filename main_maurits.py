# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 13:38:06 2021

@author: MauritsOever
"""

# packages 
# set directory...
import os
os.chdir(r"C:\Users\gebruiker\Documents\GitHub\QFRM")
import pandas as pd
import numpy as np


# import data
nikkei = pd.read_csv(r"C:\Users\gebruiker\Documents\GitHub\QFRM\NIKKEI_225.csv")
jse = pd.read_csv(r"C:\Users\gebruiker\Documents\GitHub\QFRM\JSE_TOP40.csv")
aex = pd.read_csv(r"C:\Users\gebruiker\Documents\GitHub\QFRM\AEX.csv")

# change dates to datetime, trim and set datetime as index:
nikkei['Date'] = pd.to_datetime(nikkei['Date'])
jse['Date'] = pd.to_datetime(jse['Date'])
aex['Date'] = pd.to_datetime(aex['Date'])

nikkei = pd.DataFrame(
        {'n_price': np.array(nikkei['Price']),
         'Date': np.array(nikkei['Date'])
                })
jse = pd.DataFrame(
        {'j_price': np.array(jse['Settlement Price']), #is it settlement price though?
         'Date': np.array(jse['Date'])
                })
aex = pd.DataFrame(
        {'a_price': np.array(aex['Price']),
         'Date': np.array(aex['Date'])
                })

dates = nikkei['Date']

nikkei.set_index('Date', inplace=True)
jse.set_index('Date', inplace=True)
aex.set_index('Date', inplace=True)

# create new master df with all mfing uuuuhhh price series...
df = pd.DataFrame()
df['Date'] = dates
df.set_index('Date', inplace=True)

df = pd.merge(df, nikkei, left_index = True, right_index = True)
df = pd.merge(df, jse, left_index = True, right_index = True)
df = pd.merge(df, aex, left_index = True, right_index = True)

df = df.ffill(axis=0)

# still need to get:
# - fx rates to convert everything to eur
# - from eur prices get returns
# - get some sort of debt
# - get portfoliowide return...


###############################################################################
# actual assignment part
###############################################################################



























