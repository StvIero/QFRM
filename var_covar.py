import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
#Import loss dataframe.
data = pd.read_csv('/Users/connorstevens/Documents/GitHub/qfrm_code/mv_clean_data.csv', index_col = 0)
assets = ['nikkei_ret', 'nasdaq_ret', 'aex_ret', 'eem_ret', 'dbc_ret', 'loanrate']
df = data[['nikkei_ret', 'nasdaq_ret', 'aex_ret', 'eem_ret', 'dbc_ret', 'loanrate', 'loss']].copy()

w = np.array([0.25, 0.25, 0.25, 0.25, 0.25, -0.25])
# #Pull variance forecasts out of lists.
# aex_var = np.std(df['aex_ret'])**2
# nikkei_var = np.std(df['nikkei_ret'])**2
# jse_var = np.std(df['jse_ret'])**2
# libor_var = np.std(df['libor'])**2

def varcovar(df, alpha, dist, DoF, VaRES, weights, port_val):
    # Store asset variances.
    asset_var = df.iloc[1:, 0:-1].var()

    #Create correlation matrix which will be held constant.
    #df['port_var'] = np.zeros(len(df['aex_ret']))
    port_corr = df.iloc[1:, 0:-1].corr()

    #Create covariance matrix to fill.
    port_covar = pd.DataFrame().reindex_like(port_corr)
    for j in range(0, np.shape(port_corr)[0]):
        for i in range(0, np.shape(port_corr)[0]):
            port_covar.iloc[i, j] = port_corr.iloc[i,j] * np.sqrt(asset_var.iloc[i]) * np.sqrt(asset_var.iloc[j])

    #Calculate portfolio variance.
    portvar = np.dot(weights.T, np.dot(np.array(port_covar), weights))

    mean_rets = df[assets].dropna().mean(axis=0)
    port_ret = np.dot(np.array(mean_rets), weights)
    port_vol = np.sqrt(portvar)

    # VaR normal or student-t.
    if (dist == 'normal'):
        VaR = (stats.norm.ppf(1-alpha)*port_vol - port_ret )* port_val
        #port_ret + (stats.norm.pdf(stats.norm.ppf(alpha))/(1-alpha) * port_vol) * value_port )* -1
    elif(dist == 'student-t'):
        VaR = (stats.t.ppf(1-alpha, DoF)*port_vol - port_ret) * port_val
            #(port_ret - stats.t.pdf(alpha, DoF) * port_vol) * value_port * -1
            #/(np.sqrt(DoF/(DoF - 2))))

    if (dist == 'normal'):
        ES = (alpha**-1 * stats.norm.pdf(stats.norm.ppf(alpha))*port_vol - port_ret) * port_val

    elif(dist == 'student-t'):
        xanu = stats.t.ppf(alpha, DoF)

        ES = (-1 / alpha * (1 - DoF) ** (-1) * (DoF - 2 + xanu ** 2) * stats.t.pdf(xanu, DoF) * port_vol - port_ret) * port_val
        #frac11 = (DoF + (stats.t.ppf((1- alpha), DoF))** 2) / (DoF - 1)

        #frac12 = stats.t.pdf(stats.t.ppf((1- alpha), DoF), DoF, 0, 1) / (alpha)

        #ES = (port_ret - port_vol * frac11 * frac12) * value_port * -1

    if VaRES == 'VaR':
        return (VaR)

    elif VaRES == 'ES':
        return (ES)

# print(varcovar(df, alpha = 0.01, dist = 'normal', DoF = 3.5, VaRES = 'VaR'))
#
# print(varcovar(df, alpha = 0.01, dist = 'normal', DoF = 3.5, VaRES = 'ES'))
#
# print(varcovar(df, alpha = 0.01, dist = 'student-t', DoF = 3.5, VaRES = 'VaR'))
#
# print(varcovar(df, alpha = 0.01, dist = 'student-t', DoF = 3.5, VaRES = 'ES'))


####BACKTESTING
window = 1000
var_es975VCVn = pd.DataFrame(np.zeros((np.shape(df)[0] - window, 4)))
var_es975VCVn = var_es975VCVn.rename(columns = {0: 'VaR', 1: 'ES', 2: 'Loss', 3: 'Date'})

var_es99VCVn = pd.DataFrame(np.zeros((np.shape(df)[0] - window, 4)))
var_es99VCVn = var_es99VCVn.rename(columns = {0: 'VaR', 1: 'ES', 2: 'Loss', 3: 'Date'})

var_es975VCVt = pd.DataFrame(np.zeros((np.shape(df)[0] - window, 4)))
var_es975VCVt = var_es975VCVt.rename(columns = {0: 'VaR', 1: 'ES', 2: 'Loss', 3: 'Date'})

var_es99VCVt = pd.DataFrame(np.zeros((np.shape(df)[0] - window, 4)))
var_es99VCVt = var_es99VCVt.rename(columns = {0: 'VaR', 1: 'ES', 2: 'Loss', 3: 'Date'})
# es975 = np.zeros((np.shape(df)[0] - window, 3))
# es99 = np.zeros((np.shape(df)[0] - window, 3))


for i in range(1, np.shape(df)[0] - window): #2103):
    var_es975VCVn.iloc[i, 0] = varcovar(df.iloc[i: window + i], alpha = 0.025, dist = 'normal', DoF = 3.5, VaRES = 'VaR', weights=w, port_val=100)
    var_es975VCVn.iloc[i, 1] = varcovar(df.iloc[i: window + i], alpha = 0.025, dist = 'normal', DoF = 3.5, VaRES = 'ES', weights=w, port_val=100)
    var_es975VCVn.iloc[i, 2] = df.iloc[i, -1]
    var_es975VCVn.iloc[i, 3] = df.index.values[i]

    var_es99VCVn.iloc[i, 0] = varcovar(df.iloc[i: window + i], alpha = 0.01, dist = 'normal', DoF = 3.5, VaRES = 'VaR', weights=w, port_val=100)
    var_es99VCVn.iloc[i, 1] = varcovar(df.iloc[i: window + i], alpha = 0.01, dist = 'normal', DoF = 3.5, VaRES = 'ES', weights=w, port_val=100)
    var_es99VCVn.iloc[i, 2] = df.iloc[i, -1]
    var_es99VCVn.iloc[i, 3] = df.index.values[i]

    var_es975VCVt.iloc[i, 0] = varcovar(df.iloc[i: window + i], alpha = 0.025, dist = 'student-t', DoF = 3.5, VaRES = 'VaR', weights=w, port_val=100)
    var_es975VCVt.iloc[i, 1] = varcovar(df.iloc[i: window + i], alpha = 0.025, dist = 'student-t', DoF = 3.5, VaRES = 'ES', weights=w, port_val=100)
    var_es975VCVt.iloc[i, 2] = df.iloc[i, -1]
    var_es99VCVt.iloc[i, 3] = df.index.values[i]

    var_es99VCVt.iloc[i, 0] = varcovar(df.iloc[i: window + i], alpha = 0.01, dist = 'student-t', DoF = 3.5, VaRES = 'VaR', weights=w, port_val=100)
    var_es99VCVt.iloc[i, 1] = varcovar(df.iloc[i: window + i], alpha = 0.01, dist = 'student-t', DoF = 3.5, VaRES = 'ES', weights=w, port_val=100)
    var_es99VCVt.iloc[i, 2] = df.iloc[i, -1]
    var_es99VCVt.iloc[i, 3] = df.index.values[i]


pd.DataFrame(var_es975VCVn).to_csv('/Users/connorstevens/Documents/GitHub/qfrm_code/mv/var_es975VCVn.csv')
pd.DataFrame(var_es99VCVn).to_csv('/Users/connorstevens/Documents/GitHub/qfrm_code/mv/var_es99VCVn.csv')
pd.DataFrame(var_es975VCVt).to_csv('/Users/connorstevens/Documents/GitHub/qfrm_code/mv/var_es975VCVt.csv')
pd.DataFrame(var_es99VCVt).to_csv('/Users/connorstevens/Documents/GitHub/qfrm_code/mv/var_es99VCVt.csv')


index = pd.to_datetime(df.iloc[window + 1:, :].index.values)
markers_on = var_es975VCVn['Date'][var_es975VCVn.VaR > var_es975VCVn.Loss]
plt.plot(index, var_es975VCVn.iloc[1:, 1], label = '97.5% ES')
plt.plot(index,var_es975VCVn.iloc[1:, 0], label = '97.5% VaR')
plt.plot(index,var_es975VCVn.iloc[1:, 2], alpha = 0.5, label = 'Returns')
plt.ylabel('Losses (Euros)')
plt.xlabel('Date')
plt.legend()
plt.show()