import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data.
data = pd.read_csv('/Users/connorstevens/OneDrive - Vrije Universiteit Amsterdam/QFRM/Assignments/Assignment 2/Data and Report/donald_qfrm_data.csv', index_col=0, parse_dates=True)


#Calculate prices in euros.
data['nikkei_eur'] =  data.nikkei225 * data['eur/jpy']
data['nasdaq_eur'] = data.nasdaq * data['eur/usd']
data['eem_eur'] = data.eem * data['eur/usd']
data['dbc_eur'] = data.dbc * data['eur/usd']
data['loanrate'] = data['euribor']/100 + 0.02

#Calculate log returns in euros.
data['nikkei_ret'] = (np.log(data['nikkei225']) - np.log(data['nikkei225'].shift(1)))
data['nasdaq_ret'] = (np.log(data['nasdaq']) - np.log(data['nasdaq'].shift(1)))
data['aex_ret'] = (np.log(data['aex']) - np.log(data['aex'].shift(1)))
data['eem_ret'] = (np.log(data['eem']) - np.log(data['eem'].shift(1)))
data['dbc_ret'] = (np.log(data['dbc']) - np.log(data['dbc'].shift(1)))
data['loanrate_change'] = data['loanrate'] - data['loanrate'].shift(1)

#Set initial change to zero.
data.loc[data.index.values[0], ['nikkei_ret', 'nasdaq_ret', 'aex_ret', 'eem_ret', 'dbc_ret', 'loanrate_change']] = 0

#Daily portfolio loss column.
#data['loss'] = -(25 * np.exp(data.nikkei_ret) + 25 * np.exp(data.nasdaq_ret) + 25 * np.exp(data.aex_ret) + 25 * np.exp(data.eem_ret) + 25 * np.exp(data.dbc_ret) - 25 * np.exp(data.Euribor_change))
data['loss'] = 25 * (1 - np.exp(data.aex_ret)) + 25 * (1 - np.exp(data.nikkei_ret)) + 25 * (1 - np.exp(data.nasdaq_ret))+ 25 * (1 - np.exp(data.dbc_ret)) + 25 * (1 - np.exp(data.nasdaq_ret))  - 25 * (1 - np.exp(-data.loanrate_change + 0.02/250)) + 25 * (data.loanrate + 0.02)/250
plt.plot(data.loss)
plt.show()
#data.to_csv('/Users/connorstevens/Documents/GitHub/qfrm_code/mv_clean_data.csv')

#Specify which years are to be used for VaR and ES calculations.
test = data["2013-03-25":"2015-03-17"]

#Calculte Value-at-Risk based on historical simulation.
historical_sim_VaR975 = np.quantile(test['loss'][1:], 0.975)
historical_sim_VaR99 = np.quantile(test['loss'][1:], 0.99)

#Caluclate Expected Shortfall based on historical simulation.
historical_sim_ES975 = np.mean(test['loss'][test['loss'] >= historical_sim_VaR975])
historical_sim_ES99 = np.mean(test['loss'][test['loss'] >= historical_sim_VaR99])
