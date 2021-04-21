# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 13:38:06 2021

@author: MauritsOever
"""

# packages 
# set directory...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#These two are for importing csv from github, will need to install requests, io is installed by default.
import requests
import io

# variable specification:
abs_weights = [10, 10, 10]
rel_weights = [abs_weights[0]/sum(abs_weights), abs_weights[1]/sum(abs_weights), abs_weights[2]/sum(abs_weights)]


##import data.
#Pull csv from GitHub so we dont have to keep changing directories and file paths.
Nikurl = "https://github.com/EarlGreyIsBae/QFRM/raw/main/Data/NIKKEI_225.csv"
Nikdownload = requests.get(Nikurl).content
nikkei = pd.read_csv(io.StringIO(Nikdownload.decode('utf-8')))

JSEurl = "https://github.com/EarlGreyIsBae/QFRM/raw/main/Data/JSE_TOP40.csv"
JSEdownload = requests.get(JSEurl).content
jse = pd.read_csv(io.StringIO(JSEdownload.decode('utf-8')))

AEXurl = "https://github.com/EarlGreyIsBae/QFRM/raw/main/Data/AEX.csv"
AEXdownload = requests.get(AEXurl).content
aex = pd.read_csv(io.StringIO(AEXdownload.decode('utf-8')))

LIBurl = "https://github.com/EarlGreyIsBae/QFRM/raw/main/Data/EUR3MTD156N_YMD.csv"
LIBdownload = requests.get(LIBurl).content
EUR_Libor = pd.read_csv(io.StringIO(LIBdownload.decode('utf-8')))

#FX rates.
EYurl = "https://github.com/EarlGreyIsBae/QFRM/raw/main/Data/EUR_YEN.csv"
EYdownload = requests.get(EYurl).content
euryen = pd.read_csv(io.StringIO(EYdownload.decode('utf-8')))

EZurl = "https://github.com/EarlGreyIsBae/QFRM/raw/main/Data/EUR_ZAR.csv"
EZdownload = requests.get(EZurl).content
eurzar = pd.read_csv(io.StringIO(EZdownload.decode('utf-8')))

#jse['Last'] = pd.to_numeric(jse['Last'])
# debt:

# change dates to datetime, trim and set datetime as index:
nikkei['Date'] = pd.to_datetime(nikkei['Date'])
jse['Date'] = pd.to_datetime(jse['Date'])
aex['Date'] = pd.to_datetime(aex['Date'])
# nikkei['Date2'] = nikkei['Date']
# jse['Date2'] = jse['Date']
# aex['Date2'] = aex['Date']
euryen['Date'] = pd.to_datetime(euryen['Date'])
eurzar['Date'] = pd.to_datetime(eurzar['Date'])
euryen.set_index('Date', inplace=True)
eurzar.set_index('Date', inplace=True)

dates = pd.date_range(start = "2011-03-01", end = "2021-03-01", freq="D")
#nikkei['Date']

nikkei.set_index('Date', inplace=True)
jse.set_index('Date', inplace=True)
aex.set_index('Date', inplace=True)

# nikkei = pd.merge(nikkei, euryen, left_index=True, right_index=True)
# jse = pd.merge(jse, eurzar, left_index=True, right_index=True)

# nikkei['price_eur'] = nikkei['Price']*nikkei['Mid']
# jse['price_eur'] = jse['Last']*jse['Mid']

# nikkei = pd.DataFrame(
#         {'n_price': np.array(nikkei['price_eur']),
#          'Date': np.array(nikkei['Date2'])
#                 })


# #nikkei['Date'] = pd.to_datetime(nikkei['Date']).dt.date

# jse = pd.DataFrame(
#         {'j_price': np.array(jse['price_eur']), 
#          'Date': np.array(jse['Date2'])
#                 })

# #jse['Date'] = pd.to_datetime(jse['Date']).dt.date


# aex = pd.DataFrame(
#         {'a_price': np.array(aex['Price']),
#          'Date': np.array(aex['Date2'])
#                 })

# #aex['Date'] = pd.to_datetime(aex['Date']).dt.date

# nikkei.set_index('Date', inplace=True) # kind of roundaboutish and hacky but
# jse.set_index('Date', inplace=True)    # I just want this to run properly now
# aex.set_index('Date', inplace=True) 

#Change libor data date format to match others for merge later.
EUR_Libor['Date'] = pd.to_datetime(EUR_Libor['Date'], format = "%Y/%m/%d %H:%M:%S")
EUR_Libor.set_index('Date', inplace = True)

#Rename column to something more self-explanatory.
EUR_Libor = EUR_Libor.rename(columns = {'EUR3MTD156N': '3M_EUR_Libor'})

#Change to numeric, was importing as a string.
EUR_Libor['3M_EUR_Libor'] = (pd.to_numeric(EUR_Libor['3M_EUR_Libor'], errors='coerce'))/100 #Was in percent.


#Function to replace '.' observations with average of previous and subsequent observations.
EUR_Libor['3M_EUR_Libor'] = EUR_Libor['3M_EUR_Libor'].interpolate(method = 'linear', axis = 0)



# create new master df with all mfing uuuuhhh price series...
df = pd.DataFrame()
df['Date'] = dates
df.set_index('Date', inplace=True)

#Merge all dataframes together.
df = pd.merge(df, nikkei['Price'], left_index = True, right_index = True)#Price :10754
df = pd.merge(df, jse['Last'], left_index = True, right_index = True)#Last
df = pd.merge(df, aex['Price'], left_index = True, right_index = True)# Price_y
df = pd.merge(df, EUR_Libor, left_index = True, right_index = True)# 3M_EUR_Libor
df = pd.merge(df, euryen['Bid'], left_index = True, right_index = True)#
df = pd.merge(df, eurzar['Bid'], left_index = True, right_index = True)#

#Change column names to distinguish between bid prices.
df = df.rename(columns = {'Price_x': 'nikkei'})
df = df.rename(columns = {'Last': 'jse'})
df = df.rename(columns = {'Price_y': 'aex'})
df = df.rename(columns = {'3M_EUR_Libor': 'libor'})
df = df.rename(columns = {'Bid_x': 'euryen_bid'})
df = df.rename(columns = {'Bid_y': 'eurzar_bid'})

#Fill in missing values.
df = df.ffill(axis=0)



#Get foreign prices in euros.
df['jse_eur'] = df['jse'] * df['eurzar_bid']
df['nikkei_eur'] = df['nikkei'] * df['euryen_bid']
df['jse_ret'] = np.log(df.jse_eur) - np.log(df.jse_eur.shift(1))
df['nikkei_ret'] = np.log(df.nikkei_eur) - np.log(df.nikkei_eur.shift(1))
df['aex_ret'] = np.log(df.aex) - np.log(df.aex.shift(1))

"""
Think that is all we need to calculate the portfolio losses, delete this comment if you think thats fine, add comment if you think something is missing. I would like to save this code and do our actual calculations in another file.
"""


# still need to get:
# - from eur prices get returns
# - get some sort of debt
# - get portfoliowide return...


df['Vt nikkei'] = abs_weights[0]*df['n_price']
df['Vt jse'] = abs_weights[1]*df['j_price']
df['Vt aex'] = abs_weights[2]*df['a_price']

df['Vt'] = df['Vt nikkei'] + df['Vt jse'] + df['Vt aex']
df['Vt_ret'] = np.log(df.Vt) - np.log(df.Vt.shift(1))
df['nik_ret'] = np.log(df.n_price) - np.log(df.n_price.shift(1))
df['jse_ret'] = np.log(df.j_price) - np.log(df.j_price.shift(1))
df['aex_ret'] = np.log(df.a_price) - np.log(df.a_price.shift(1))


###############################################################################
# actual assignment part
###############################################################################
df = df.iloc[1:]

# var-covar on multivariate normal dist:
wvol_n = rel_weights[0]**2 * np.std(df.nik_ret) 
wvol_j = rel_weights[1]**2 * np.std(df.jse_ret) 
wvol_a = rel_weights[2]**2 * np.std(df.aex_ret) 

wcov_nj = rel_weights[0]*rel_weights[1]*np.cov(df.nik_ret, df.jse_ret)[0,1]
wcov_na = rel_weights[0]*rel_weights[2]*np.cov(df.nik_ret, df.aex_ret)[0,1]
wcov_ja = rel_weights[1]*rel_weights[2]*np.cov(df.jse_ret, df.aex_ret)[0,1]





















