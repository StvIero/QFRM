import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
#Import loss dataframe.
data = pd.read_csv('/Users/connorstevens/Documents/GitHub/QFRM/Data/loss_df.csv', index_col= 0)

df = data[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor', 'loss']].copy()

# #Pull variance forecasts out of lists.
# aex_var = np.std(df['aex_ret'])**2
# nikkei_var = np.std(df['nikkei_ret'])**2
# jse_var = np.std(df['jse_ret'])**2
# libor_var = np.std(df['libor'])**2

#Store asset variances.
asset_var = np.array([aex_var, nikkei_var, jse_var, libor_var])

def varcovar(df, alpha, dist, DoF, VaRES):
    # Pull variance forecasts out of lists.
    aex_var = np.std(df['aex_ret']) ** 2
    nikkei_var = np.std(df['nikkei_ret']) ** 2
    jse_var = np.std(df['jse_ret']) ** 2
    libor_var = np.std(df['libor']) ** 2

    # Store asset variances.
    asset_var = np.array([aex_var, nikkei_var, jse_var, libor_var])

    #Create correlation matrix which will be held constant.
    #df['port_var'] = np.zeros(len(df['aex_ret']))
    port_corr = np.array(df[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor']].corr())
    weights = np.array([0.6, 0.6, 0.3, -0.5])

    #Create covariance matrix to fill.
    port_covar = np.zeros((4,4))
    for j in range(0, 3):
        for i in range(0, 3):
            port_covar[i, j] = port_corr[i,j] * np.sqrt(asset_var[i]) * np.sqrt(asset_var[j])

    #Calculate portfolio variance.
    portvar = np.dot(weights.T, np.dot(port_covar, weights))

    mean_rets = np.array(df[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor']].dropna().mean(axis=0))
    port_ret = np.dot(mean_rets, weights)
    port_vol = np.sqrt(portvar)

    # VaR normal or student-t.
    value_port = 100_000_000
    if (dist == 'normal'):
        VaR = (stats.norm.ppf(1-alpha)*port_vol - port_ret )* 100000000
        #port_ret + (stats.norm.pdf(stats.norm.ppf(alpha))/(1-alpha) * port_vol) * value_port )* -1
    elif(dist == 'student-t'):
        VaR = (stats.t.ppf(1-alpha, DoF)*port_vol - port_ret) * 100000000
            #(port_ret - stats.t.pdf(alpha, DoF) * port_vol) * value_port * -1
            #/(np.sqrt(DoF/(DoF - 2))))

    if (dist == 'normal'):
        ES = (alpha**-1 * stats.norm.pdf(stats.norm.ppf(alpha))*port_vol - port_ret) * 100000000

    elif(dist == 'student-t'):
        xanu = stats.t.ppf(alpha, DoF)

        ES = (-1 / alpha * (1 - DoF) ** (-1) * (DoF - 2 + xanu ** 2) * stats.t.pdf(xanu, DoF) * port_vol - port_ret) * 100000000
        #frac11 = (DoF + (stats.t.ppf((1- alpha), DoF))** 2) / (DoF - 1)

        #frac12 = stats.t.pdf(stats.t.ppf((1- alpha), DoF), DoF, 0, 1) / (alpha)

        #ES = (port_ret - port_vol * frac11 * frac12) * value_port * -1

    if VaRES == 'VaR':
        return (VaR)

    elif VaRES == 'ES':
        return (ES)

print(varcovar(df, alpha = 0.01, dist = 'normal', DoF = 3.5, VaRES = 'VaR'))

print(varcovar(df, alpha = 0.01, dist = 'normal', DoF = 3.5, VaRES = 'ES'))

print(varcovar(df, alpha = 0.01, dist = 'student-t', DoF = 3.5, VaRES = 'VaR'))

print(varcovar(df, alpha = 0.01, dist = 'student-t', DoF = 3.5, VaRES = 'ES'))


####BACKTESTING
window = 1000
var_es975VCVn = np.zeros((np.shape(df)[0] - window, 3))
var_es99VCVn = np.zeros((np.shape(df)[0] - window, 3))
var_es975VCVt = np.zeros((np.shape(df)[0] - window, 3))
var_es99VCVt = np.zeros((np.shape(df)[0] - window, 3))
# es975 = np.zeros((np.shape(df)[0] - window, 3))
# es99 = np.zeros((np.shape(df)[0] - window, 3))


for i in range(1, np.shape(df)[0] - window): #2103):
    var_es975VCVn[i, 0] = varcovar(df[i: window + i], alpha = 0.025, dist = 'normal', DoF = 3.5, VaRES = 'VaR')
    var_es975VCVn[i, 1] = varcovar(df[i: window + i], alpha = 0.025, dist = 'normal', DoF = 3.5, VaRES = 'ES')
    var_es975VCVn[i, 2] = df.iloc[i, -1]

    var_es99VCVn[i, 0] = varcovar(df[i: window + i], alpha = 0.01, dist = 'normal', DoF = 3.5, VaRES = 'VaR')
    var_es99VCVn[i, 1] = varcovar(df[i: window + i], alpha = 0.01, dist = 'normal', DoF = 3.5, VaRES = 'ES')
    var_es99VCVn[i, 2] = df.iloc[i, -1]

    var_es975VCVt[i, 0] = varcovar(df[i: window + i], alpha = 0.025, dist = 'student-t', DoF = 3.5, VaRES = 'VaR')
    var_es975VCVt[i, 1] = varcovar(df[i: window + i], alpha = 0.025, dist = 'student-t', DoF = 3.5, VaRES = 'ES')
    var_es975VCVt[i, 2] = df.iloc[i, -1]

    var_es99VCVt[i, 0] = varcovar(df[i: window + i], alpha = 0.01, dist = 'student-t', DoF = 3.5, VaRES = 'VaR')
    var_es99VCVt[i, 1] = varcovar(df[i: window + i], alpha = 0.01, dist = 'student-t', DoF = 3.5, VaRES = 'ES')
    var_es99VCVt[i, 2] = df.iloc[i, -1]


pd.DataFrame(var_es975VCVn).to_csv('/Users/connorstevens/Documents/GitHub/QFRM/Plots/var_es975VCVn.csv')
pd.DataFrame(var_es99VCVn).to_csv('/Users/connorstevens/Documents/GitHub/QFRM/Plots/var_es99VCVn.csv')
pd.DataFrame(var_es975VCVt).to_csv('/Users/connorstevens/Documents/GitHub/QFRM/Plots/var_es975VCVt.csv')
pd.DataFrame(var_es99VCVt).to_csv('/Users/connorstevens/Documents/GitHub/QFRM/Plots/var_es99VCVt.csv')


index = pd.to_datetime(df.iloc[window + 1:, :].index.values)
plt.plot(index, var_es975VCVn[1:, 1], label = '97.5% ES')
plt.plot(index,var_es975VCVn[1:, 0], label = '97.5% VaR')
plt.plot(index,var_es975VCVn[1:, 2], alpha = 0.5, label = 'Returns')
plt.ylabel('Losses (Euros)')
plt.xlabel('Date')
plt.legend()
plt.show()