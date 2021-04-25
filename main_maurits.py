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



# change dates to datetime.
nikkei['Date'] = pd.to_datetime(nikkei['Date'])
jse['Date'] = pd.to_datetime(jse['Date'])
aex['Date'] = pd.to_datetime(aex['Date'])
euryen['Date'] = pd.to_datetime(euryen['Date'])
eurzar['Date'] = pd.to_datetime(eurzar['Date'])

#Create data series to reference in dataframe creation later.
dates = pd.date_range(start = "2011-03-01", end = "2021-03-01", freq="D")

#Set date as index
nikkei.set_index('Date', inplace=True)
jse.set_index('Date', inplace=True)
aex.set_index('Date', inplace=True)
euryen.set_index('Date', inplace=True)
eurzar.set_index('Date', inplace=True)

#Change libor data date format to match others for merge later.
EUR_Libor['Date'] = pd.to_datetime(EUR_Libor['Date'], format = "%Y/%m/%d %H:%M:%S")
EUR_Libor.set_index('Date', inplace = True)

#Rename column to something more self-explanatory.
EUR_Libor = EUR_Libor.rename(columns = {'EUR3MTD156N': '3M_EUR_Libor'})

#Change to numeric, was importing as a string.
EUR_Libor['3M_EUR_Libor'] = (pd.to_numeric(EUR_Libor['3M_EUR_Libor'], errors='coerce'))/100 #Was in percent.


#Function to replace '.' observations with average of previous and subsequent observations.
EUR_Libor['3M_EUR_Libor'] = EUR_Libor['3M_EUR_Libor'].interpolate(method = 'linear', axis = 0)



#Empty datafram to store merged data.
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



#Get foreign prices in euros and changes in interest rate.
df['jse_eur'] = df['jse'] * df['eurzar_bid']
df['nikkei_eur'] = df['nikkei'] * df['euryen_bid']
df['jse_ret'] = np.log(df.jse_eur) - np.log(df.jse_eur.shift(1))
df['nikkei_ret'] = np.log(df.nikkei_eur) - np.log(df.nikkei_eur.shift(1))
df['aex_ret'] = np.log(df.aex) - np.log(df.aex.shift(1))
df['libor_change'] = (df.libor - df.libor.shift(1))/100

#Set portfolio absolute weights
port_val = 100_000_000
aex_val = 0.6 * port_val
nikkei_val = 0.6 * port_val
jse_val = 0.3 * port_val
#Short debt, therefore weight is actually negative.
debt_val = 0.5 * port_val

df['loss'] = aex_val * (1 - np.exp(df.aex_ret)) + nikkei_val * (1 - np.exp(df.nikkei_ret)) + jse_val * (1 - np.exp(df.jse_ret))  - debt_val * (1 - np.exp(-df.libor_change)) + debt_val * df.libor/250
#
# """
# Rebalancing Code:
# ----------------
# 100m euros:
#     50m cash
#     50m debt
#
# Weights: Relative
#     40% AEX
#     40% Nikkei
#     20% JSE
#
# """
# initial_val = 100000000
# debt_weight = -0.5
# debt_val = initial_val * debt_weight
# aex_weight = 0.6
# nikkei_weight = 0.6
# jse_weight = 0.3
#
# ##Add debt into df
#
# #Calculate losses due to changes in libor rate.
# df['debt_loss'] = -(debt_val * (1 - np.exp(-df['libor_change']))) + (debt_val * df['libor'])/250
#
#
# #Create dataframe to store data used for rebalancing calculations.
# df_re = pd.DataFrame({'aex_units': np.zeros(np.shape(df)[0] + 1),
#                                'nikkei_units': np.zeros(np.shape(df)[0] + 1),
#                                'jse_units': np.zeros(np.shape(df)[0] + 1),
#                                'aex_pos_val': np.zeros(np.shape(df)[0] + 1),
#                                'nikkei_pos_val': np.zeros(np.shape(df)[0] + 1),
#                                'jse_pos_val': np.zeros(np.shape(df)[0] + 1),
#                                'equity_val': np.zeros(np.shape(df)[0] + 1)})
#
# #Variables to reference units, position, price columns and weights.
# units = ['aex_units', 'nikkei_units', 'jse_units']
# position = ['aex_pos_val', 'nikkei_pos_val', 'jse_pos_value']
# weights = np.array([0.4, 0.4, 0.2])
# prices = ['aex', 'nikkei_eur', 'jse_eur']
#
# #Set up initial portfolio positions.
# df_re.loc[0: 1, 'aex_units'] = initial_val * aex_weight / df['aex'][0]
# df_re.loc[0, 'aex_pos_val'] = aex_weight * initial_val
#
# df_re.loc[0: 1, 'nikkei_units'] = initial_val * nikkei_weight / df['nikkei_eur'][0]
# df_re.loc[0, 'nikkei_pos_val'] = nikkei_weight * initial_val
#
# df_re.loc[0: 1, 'jse_units'] = initial_val * jse_weight / df['jse_eur'][0]
# df_re.loc[0, 'jse_pos_val'] = jse_weight * initial_val
#
# df_re.loc[0, 'equity_val'] = np.sum(df_re.iloc[0, 3:6])
#
# ####Rebalancing loop.

# for i in range(1, np.shape(df_re)[0] - 1):
# #Calculate position values.
#     df_re.loc[i, 'aex_pos_val'] = df['aex'][i] * df_re['aex_units'][i]
#     df_re.loc[i, 'nikkei_pos_val'] = df['nikkei_eur'][i] * df_re['nikkei_units'][i]
#     df_re.loc[i, 'jse_pos_val'] = df['jse_eur'][i] * df_re['jse_units'][i]
#
#     df_re.loc[i, 'equity_val'] = np.sum(df_re.iloc[i, 3:6])
#
# ###Calculate new unit numbers due to rebalancing.
#
#     #Calculate new number of units by dividing previous  weight of previous portfolio value by new price.
#     df_re.loc[i + 1, 'aex_units'] = df_re.loc[i, 'equity_val'] * aex_weight / df['aex'][i]
#     df_re.loc[i + 1, 'nikkei_units'] = df_re.loc[i, 'equity_val'] * nikkei_weight / df['nikkei_eur'][i]
#     df_re.loc[i + 1, 'jse_units'] = df_re.loc[i, 'equity_val'] * jse_weight / df['jse_eur'][i]
#
# #Not ideal, but I had to do the last line this way to get it to work.
# df_re.iloc[-1, df_re.columns.get_loc('aex_pos_val')] = df.aex.iloc[-1] * df_re.aex_units.iloc[-1]
# df_re.iloc[-1, df_re.columns.get_loc('nikkei_pos_val')] = df.nikkei_eur.iloc[-1] * df_re.nikkei_units.iloc[-1]
# df_re.iloc[-1, df_re.columns.get_loc('jse_pos_val')] = df.jse_eur.iloc[-1] * df_re.jse_units.iloc[-1]
#
# index_aex = df_re.columns.get_loc('aex_pos_val')
# index_nikkei = df_re.columns.get_loc('nikkei_pos_val')
# index_jse = df_re.columns.get_loc('jse_pos_val')
#
# df_re.iloc[-1, -1] = np.sum(df_re.iloc[-1, [index_aex, index_nikkei, index_jse]])
#
# #Add equity_val to df. First line exluded because used as a starting point for positons.
# df['equity_val'] = np.array(df_re.loc[1:, 'equity_val'])
#
# #Calculate equity returns.
# df['equity_ret'] = np.log(df.equity_val) - np.log(df.equity_val.shift(1))
#
# #Calculate equity losses.
# df['equity_loss'] = np.zeros(np.shape(df)[0])
#
# #First loss done manually.
# df.iloc[0, df.columns.get_loc('equity_loss')] = -(df.iloc[0, df.columns.get_loc('equity_val')] - initial_val)
#
# for i in range(1, np.shape(df)[0]):
#     df.iloc[i, df.columns.get_loc('equity_loss')] = (df.iloc[i, df.columns.get_loc('equity_val')] - df.iloc[i - 1, df.columns.get_loc('equity_val')])
#
#
# #Daily portfolio losses.
# df['total_loss'] = df['equity_loss'] + df['debt_loss']
# df['total_loss'] = df['equity_loss'] + df['debt_loss']

#df.to_csv('/Users/connorstevens/Documents/GitHub/QFRM/Data/loss_df.csv')

####BACKTESTING
var975 = np.zeros(2103)
var99 = np.zeros(2103)
es975 = np.zeros(2103)
es99 = np.zeros(2103)



len(df['aex_ret'].dropna())
for i in range(1, 2103):
    var975[i] = np.quantile(df.iloc[i: i + 250, -1], 0.975)
    var99[i] = np.quantile(df.iloc[i: i + 250, -1], 0.99)


#Calculte Value-at-Risk based on historical simulation.
historical_sim_VaR975 = np.quantile(df['loss'][1:], 0.975)
historical_sim_VaR99 = np.quantile(df['loss'][1:], 0.99)

#Caluclate Expected Shortfall based on historical simulation.
historical_sim_ES975 = np.mean(df['loss'][df['loss'] >= historical_sim_VaR975])
historical_sim_ES99 = np.mean(df['loss'][df['loss'] >= historical_sim_VaR99])

#Print results.
print('97.5% VaR is ' + str(round(historical_sim_VaR975, 2)))
print('97.5% ES is ' + str(round(historical_sim_ES975, 2)))
print('99% VaR is ' + str(round(historical_sim_VaR99, 2)))
print('99% ES is ' + str(round(historical_sim_ES99, 2)))

sns.histplot(df.loss, kde = True)
plt.show()