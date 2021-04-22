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
from scipy.stats import norm
from scipy.stats import t
import statsmodels.api as sm
#These two are for importing csv from github, will need to install requests, io is installed by default.
import requests
import io

# variable specification:
abs_weights = [40, 40, 20]
rel_weights = [abs_weights[0]/sum(abs_weights), abs_weights[1]/sum(abs_weights), abs_weights[2]/sum(abs_weights)]
port_value = 100000000


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
Rebalancing Code:
----------------
100m euros:
    50m cash
    50m debt
Weights: Relative
    40% AEX
    40% Nikkei
    20% JSE
    
"""
initial_val = 100000000
debt_weight = 0.5
debt_val = initial_val * debt_weight
aex_weight = 0.4
nikkei_weight = 0.4
jse_weight = 0.2

##Add debt into df.

#Change in euro libor rate.
df['libor_change'] = df.libor - df.libor.shift(1)

#Calculate losses due to changes in libor rate.
#df['debt_return'] =


#Create dataframe to store data used for rebalancing calculations.
df_re = pd.DataFrame({'aex_units': np.zeros(np.shape(df)[0] + 1),
                               'nikkei_units': np.zeros(np.shape(df)[0] + 1),
                               'jse_units': np.zeros(np.shape(df)[0] + 1),
                               'aex_pos_val': np.zeros(np.shape(df)[0] + 1),
                               'nikkei_pos_val': np.zeros(np.shape(df)[0] + 1),
                               'jse_pos_val': np.zeros(np.shape(df)[0] + 1),
                               'equity_val': np.zeros(np.shape(df)[0] + 1)})

#Variables to reference units, position, price columns and weights.
units = ['aex_units', 'nikkei_units', 'jse_units']
position = ['aex_pos_val', 'nikkei_pos_val', 'jse_pos_value']
weights = np.array([0.4, 0.4, 0.2])
prices = ['aex', 'nikkei_eur', 'jse_eur']

#Set up initial portfolio positions.
df_re.loc[0: 1, 'aex_units'] = initial_val * aex_weight / df['aex'][0]
df_re.loc[0, 'aex_pos_val'] = aex_weight * initial_val

df_re.loc[0: 1, 'nikkei_units'] = initial_val * nikkei_weight / df['nikkei_eur'][0]
df_re.loc[0, 'nikkei_pos_val'] = nikkei_weight * initial_val

df_re.loc[0: 1, 'jse_units'] = initial_val * jse_weight / df['jse_eur'][0]
df_re.loc[0, 'jse_pos_val'] = jse_weight * initial_val

df_re.loc[0, 'equity_val'] = np.sum(df_re.iloc[0, 3:6])

####Rebalancing loop.

for i in range(1, np.shape(df_re)[0] - 1):
#Calculate position values.
    df_re.loc[i, 'aex_pos_val'] = df['aex'][i] * df_re['aex_units'][i]
    df_re.loc[i, 'nikkei_pos_val'] = df['nikkei_eur'][i] * df_re['nikkei_units'][i]
    df_re.loc[i, 'jse_pos_val'] = df['jse_eur'][i] * df_re['jse_units'][i]

    df_re.loc[i, 'equity_val'] = np.sum(df_re.iloc[i, 3:6])

###Calculate new unit numbers due to rebalancing.

    #Calculate new number of units by dividing previous  weight of previous portfolio value by new price.
    df_re.loc[i + 1, 'aex_units'] = df_re.loc[i, 'equity_val'] * aex_weight / df['aex'][i]
    df_re.loc[i + 1, 'nikkei_units'] = df_re.loc[i, 'equity_val'] * nikkei_weight / df['nikkei_eur'][i]
    df_re.loc[i + 1, 'jse_units'] = df_re.loc[i, 'equity_val'] * jse_weight / df['jse_eur'][i]

#Not ideal, but I had to do the last line this way to get it to work.
df_re.iloc[-1, df_re.columns.get_loc('aex_pos_val')] = df.aex.iloc[-1] * df_re.aex_units.iloc[-1]
df_re.iloc[-1, df_re.columns.get_loc('nikkei_pos_val')] = df.nikkei_eur.iloc[-1] * df_re.nikkei_units.iloc[-1]
df_re.iloc[-1, df_re.columns.get_loc('jse_pos_val')] = df.jse_eur.iloc[-1] * df_re.jse_units.iloc[-1]

index_aex = df_re.columns.get_loc('aex_pos_val')
index_nikkei = df_re.columns.get_loc('nikkei_pos_val')
index_jse = df_re.columns.get_loc('jse_pos_val')

df_re.iloc[-1, -1] = np.sum(df_re.iloc[-1, [index_aex, index_nikkei, index_jse]])

#Add equity_val to df. First line exluded because used as a starting point for positons.
df['equity_val'] = np.array(df_re.loc[1:, 'equity_val'])

#Calculate equity returns.
df['equity_ret'] = np.log(df.equity_val) - np.log(df.equity_val.shift(1))


"""
End of rebalancing code
"""



"""
###############################################################################
# actual assignment part
###############################################################################
"""

def ST_VAR_ES(nu, SD_port):
    sigma = SD_port / np.sqrt(nu/(nu-2))
    VaR975 = (average_port_ret + t.ppf(0.025, nu, 0, 1)*sigma)*port_value*-1
    VaR990 = (average_port_ret + t.ppf(0.01, nu, 0, 1)*sigma)*port_value*-1
    
    frac11 = (nu+(t.ppf(0.025, nu, 0, 1))**2)/(nu-1)
    frac12 = t.pdf(t.ppf(0.025, nu, 0, 1), nu, 0, 1)/(0.025)
    ES975 = (average_port_ret - sigma*frac11*frac12)*port_value*-1
    
    frac21 = (nu+(t.ppf(0.01, nu, 0, 1))**2)/(nu-1)
    frac22 = t.pdf(t.ppf(0.01, nu, 0, 1), nu, 0, 1)/(0.01)
    ES990 = (average_port_ret - sigma*frac21*frac22)*port_value*-1

    print('97.5% VaR is', VaR975)
    print('99.0% VaR is', VaR990)
    print('')
    print('97.5% ES is', ES975)
    print('99.0% ES is', ES990)
    print('')
    print('')
    
    return

df = df.iloc[1:]
average_port_ret = np.mean(rel_weights[0]*df.nikkei_ret + rel_weights[1]*df.aex_ret + rel_weights[2]*df.jse_ret)

# var-covar on multivariate normal dist:
wvol_n = rel_weights[0]**2 * np.std(df.nikkei_ret)**2
wvol_j = rel_weights[1]**2 * np.std(df.jse_ret)**2 
wvol_a = rel_weights[2]**2 * np.std(df.aex_ret)**2 

wcov_nj = rel_weights[0]*rel_weights[1]*np.cov(df.nikkei_ret, df.jse_ret)[0,1]
wcov_na = rel_weights[0]*rel_weights[2]*np.cov(df.nikkei_ret, df.aex_ret)[0,1]
wcov_ja = rel_weights[1]*rel_weights[2]*np.cov(df.jse_ret, df.aex_ret)[0,1]

# get portfolio vol, to get VaR:
vol_port = np.sqrt(wvol_a + wvol_j + wvol_n + wcov_nj + wcov_na + wcov_ja)

# normal VaRs 
print('Assuming rets are normal:')
print('97.5% VaR is', (average_port_ret-1.96*vol_port)*port_value*-1) # 1,884,792
print('99.0% VaR is', (average_port_ret -2.36*vol_port)*port_value*-1) #2,274,832
print('')
# normal ES's
# ES formula = pdf(cdfinv(alpha))/(alpha) * sigma

print('97.5% ES is', (average_port_ret*-1 + norm.pdf(norm.ppf(0.025))/(0.025) * vol_port) * port_value)
print('99.0% ES is', (average_port_ret*-1 + norm.pdf(norm.ppf(0.01))/(0.01) * vol_port) * port_value)
print('')

# now for student t holmes:

for i in [3,4,5,6]:
    print('If nu =', i)
    ST_VAR_ES(i, vol_port)

# get QQ-plot

sm.qqplot(df['equity_ret']/np.std(df['equity_ret']), line='45')



