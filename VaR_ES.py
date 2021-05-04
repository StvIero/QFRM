#Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats

#Import loss dataframe.
data = pd.read_csv('/Users/connorstevens/Documents/GitHub/QFRM/Data/loss_df.csv', index_col= 0)

df = data[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor', 'loss']].dropna().copy()


#Make sure start date 2011-03-02 or later. End date cannot be after 2021-03-01.
#Make sure dates are in quotations.
def CCC(df, alpha, dist, DoF, VaRES):

#Multiplied all returns by 100 due to errors form GARCH function.
    #Fit AEX GARCH model.
    garch_aex = arch_model(df['aex_ret'] * 100, vol='garch', p=1, o=0, q=1, dist='Normal', mean = 'Zero')
    aex_garch_fitted = garch_aex.fit(update_freq=5, disp="off")

    #Fit Nikkei GARCH model.
    garch_nikkei = arch_model(df['nikkei_ret'] * 100, vol='garch', p=1, o=0, q=1, dist='Normal', mean = 'Zero')
    nikkei_garch_fitted = garch_nikkei.fit(update_freq=5, disp="off")

    #Fit JSE GARCH model.
    garch_jse = arch_model(df['jse_ret'] * 100, vol='garch', p=1, o=0, q=1, dist = 'Normal', mean = 'Zero')
    jse_garch_fitted = garch_jse.fit(disp="off")

    ##Fit LIBOR GARCH model.
    garch_libor = arch_model(df['libor'] * 100, vol='garch', p=1, o=0, q=1, mean='Zero')
    libor_garch_fitted = garch_libor.fit(disp="off")

    #Forecast variance.
    aex_var_for = aex_garch_fitted.forecast(horizon= 1, reindex= False)

    nikkei_var_for = nikkei_garch_fitted.forecast(horizon= 1,reindex= False)

    jse_var_for = jse_garch_fitted.forecast(horizon= 1,reindex= False)

    libor_var_for = libor_garch_fitted.forecast(horizon= 1,reindex= False)

    #Pull variance forecasts out of lists.
    aex_var = aex_var_for.variance.values[0]
    nikkei_var = nikkei_var_for.variance.values[0]
    jse_var = jse_var_for.variance.values[0]
    libor_var = libor_var_for.variance.values[0]

    #Store asset variances.
    asset_var = np.array([aex_var, nikkei_var, jse_var, libor_var])


    #Create correlation matrix which will be held constant.
    #df['port_var'] = np.zeros(len(df['aex_ret']))
    port_corr = np.array(df[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor']].corr())
    weights = np.array([0.6, 0.6, 0.3, -0.5])

    #Create covariance matrix to fill.
    port_covar = np.zeros((4,4))
    for j in range(0, 3):
        for i in range(0, 3):
            port_covar[i, j] = port_corr[i,j] * asset_var[i] * asset_var[j]

    #Calculate portfolio variance.
    portvar = np.dot(weights.T, np.dot(port_covar, weights))

    mean_rets = np.array(df[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor']].dropna().mean(axis=0))
    port_ret = np.dot(mean_rets, weights)
    port_vol = np.sqrt(portvar)/100

    # VaR normal or student-t.
    value_port = 100_000_000
    if (dist == 'normal'):
        VaR = (stats.norm.ppf(1-alpha)*port_vol - port_ret )* 100000000
    elif(dist == 'student-t'):
        VaR = (stats.t.ppf(1-alpha, DoF)*port_vol - port_ret) * 100000000

    if (dist == 'normal'):
        ES = (alpha**-1 * stats.norm.pdf(stats.norm.ppf(alpha))*port_vol - port_ret) * 100000000

    elif(dist == 'student-t'):
        xanu = stats.t.ppf(alpha, DoF)

        ES = (-1 / alpha * (1 - DoF) ** (-1) * (DoF - 2 + xanu ** 2) * stats.t.pdf(xanu,DoF) * port_vol - port_ret) * 100000000

    if VaRES == 'VaR':
        return (VaR)

    elif VaRES == 'ES':
        return (ES)

#Print out VaR and ES values.
print('97.5% normal VaR is ' + str(round(CCC(df, alpha = 0.025, dist = 'normal', DoF = 3.5, VaRES = 'VaR'),2)))
print('97.5% normal ES is ' + str(round(CCC(df, alpha = 0.025, dist = 'normal', DoF = 3.5, VaRES = 'ES'), 2)))

print('99% normal VaR is ' + str(round(CCC(df, alpha = 0.01, dist = 'normal', DoF = 3.5, VaRES = 'VaR'), 2)))
print('99% normal ES is ' + str(round(CCC(df, alpha = 0.01, dist = 'normal', DoF = 3.5, VaRES = 'ES'), 2)))

print('97.5% student-t VaR is ' + str(round(CCC(df, alpha = 0.025, dist = 'student-t', DoF = 3.5, VaRES = 'VaR'), 2)))
print('97.5% student-t ES is ' + str(round(CCC(df, alpha = 0.025, dist = 'student-t', DoF = 3.5, VaRES = 'ES'), 2)))

print('99% student-t VaR is ' + str(round(CCC(df, alpha = 0.01, dist = 'student-t', DoF = 3.5, VaRES = 'VaR'), 2)))
print('99% student-t ES is ' + str(round(CCC(df, alpha = 0.01, dist = 'student-t', DoF = 3.5, VaRES = 'ES'), 2)))

####BACKTESTING
window = 1000
var_es975CCCn = np.zeros((np.shape(df)[0] - window, 3))
#var_es99CCCn = np.zeros((np.shape(df)[0] - window, 3))
var_es975CCCt = np.zeros((np.shape(df)[0] - window, 3))
#var_es99CCCt = np.zeros((np.shape(df)[0] - window, 3))

# es975 = np.zeros((np.shape(df)[0] - window, 3))
# es99 = np.zeros((np.shape(df)[0] - window, 3))


for i in range(1, np.shape(df)[0] - window): #2103):
    # var_es975CCCn[i, 0] = CCC(df[i: i + window ], alpha = (1-0.975), dist = 'normal', DoF = 0, VaRES= 'VaR')
    # var_es975CCCn[i, 1] = CCC(df[i: i + window ], alpha = (1-0.975), dist = 'normal', DoF = 0, VaRES= 'ES')
    # var_es975CCCn[i, 2] = df.iloc[i, -1]

    var_es99CCCn[i, 0] = CCC(df[i: i + window ], alpha = (1 -0.99), dist = 'normal', DoF = 0, VaRES= 'VaR')
    var_es99CCCn[i, 1] = CCC(df[i: i + window ], alpha = (1-0.99), dist = 'normal', DoF = 0, VaRES= 'ES')
    var_es99CCCn[i, 2] = df.iloc[i, -1]

    # var_es975CCCt[i, 0] = CCC(df[i: i + window ], alpha = (1-0.975), dist = 'student-t', DoF = 3.5, VaRES= 'VaR')
    # var_es975CCCt[i, 1] = CCC(df[i: i + window ], alpha = (1-0.975), dist = 'student-t', DoF = 3.5, VaRES= 'ES')
    # var_es975CCCt[i, 2] = df.iloc[i, -1]

    # var_es99CCCt[i, 0] = CCC(df[i: i + window ], alpha = 0.99, dist = 'student-t', DoF = 3.5, VaRES= 'VaR')
    # var_es99CCCt[i, 1] = CCC(df[i: i + window ], alpha = 0.99, dist = 'student-t', DoF = 3.5, VaRES= 'ES')
    # var_es99CCCt[i, 2] = df.iloc[i, -1]

#plt.plot(var_es975[1:, 0], label = '97.5% VaR')
# plt.plot(var_es975[1: 1], alpha = 0.5, label = '97.5% ES')
# plt.plot(var_es975[1:, 2], alpha = 0.7, label = 'Returns')
# plt.legend()
pd.to_datetime(df.index.values)
# plt.show()
window_start_index = np.shape(df)[0] - window

pd.DataFrame(var_es975CCCn).to_csv('/Users/connorstevens/Documents/GitHub/QFRM/Plots/var_es975CCCn.csv')
pd.DataFrame(var_es99CCCn).to_csv('/Users/connorstevens/Documents/GitHub/QFRM/Plots/var_es99CCCn.csv')
# pd.DataFrame(var_es975CCCt).to_csv('/Users/connorstevens/Documents/GitHub/QFRM/Plots/var_es975CCCt.csv')
# pd.DataFrame(var_es99CCCt).to_csv('/Users/connorstevens/Documents/GitHub/QFRM/Plots/var_es99CCCt.csv')

index = pd.to_datetime(df.iloc[window + 1:, :].index.values)
plt.plot(index, var_es975CCCn[1:-1, 0], label = '97.5% VaR')
plt.plot(index,var_es975CCCn[1:-1, 1], label = '97.5% ES')
plt.plot(index,var_es975CCCn[1:-1, 2], alpha = 0.5, label = 'Returns')
plt.ylabel('Losses (Euros)')
plt.xlabel('Date')
plt.legend()
plt.show()

index = pd.to_datetime(df.iloc[window + 1:, :].index.values)
plt.plot(index, var_es99CCCn[1:-1, 0], label = '99% VaR')
plt.plot(index,var_es99CCCn[1:-1, 1], label = '99% ES')
plt.plot(index,var_es99CCCn[1:-1, 2], alpha = 0.5, label = 'Returns')
plt.ylabel('Losses (Euros)')
plt.xlabel('Date')
plt.legend()
plt.show()


CCC(df, alpha = 0.975, dist = 'normal', DoF = 0)


plt.plot(aex_var_for.variance.values, label = 'AEX forecast variance', alpha = 0.5)
plt.plot(nikkei_var_for.variance.values, label = 'Nikkei forecast variance', alpha = 0.5)
plt.plot(jse_var_for.variance.values, label = 'JSE forecast variance', alpha = 0.5)
plt.plot(libor_var_for.variance.values, label = 'LIBOR forecast variance', alpha = 0.5)
plt.plot(df['port_var'], label = 'Porfolio Variance', alpha = 0.5)
#plt.plot(df['aex_ret'].dropna() * 100, alpha = 0.5, label = 'returns')
plt.legend()
plt.show()

plt.plot(df['port_var'], label = 'Porfolio Variance')
plt.legend()
plt.show()

plt.plot(df[['aex', 'jse_eur', 'nikkei_eur']], label = ['aex', 'jse', 'nikkei'])
plt.legend()
plt.show()