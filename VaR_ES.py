#Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats

#Make sure start date 2011-03-02 or later. End date cannot be after 2021-03-01.
#Make sure dates are in quotations.
def CCC(df, start, stop, alpha, dist, DoF):
    #Import loss dataframe.
    df = pd.read_csv('/Users/connorstevens/Documents/GitHub/QFRM/Data/loss_df.csv', index_col= 0)

#Multiplied all returns by 100 due to errors form GARCH function.
    #Fit AEX GARCH model.
    garch_aex = arch_model(df['aex_ret'][start:stop] * 100, vol='garch', p=1, o=0, q=1, dist='Normal', mean = 'Zero')
    aex_garch_fitted = garch_aex.fit(update_freq=5)

    #Fit Nikkei GARCH model.
    garch_nikkei = arch_model(df['nikkei_ret'][start:stop] * 100, vol='garch', p=1, o=0, q=1, dist='Normal', mean = 'Zero')
    nikkei_garch_fitted = garch_nikkei.fit(update_freq=5)

    #Fit JSE GARCH model.
    garch_jse = arch_model(df['jse_ret'][start:stop] * 100, vol='garch', p=1, o=0, q=1, dist = 'Normal', mean = 'Zero')
    jse_garch_fitted = garch_jse.fit()

    ##Fit LIBOR GARCH model.
    garch_libor = arch_model(df['libor'][start:stop] * 100, vol='garch', p=1, o=0, q=1, mean='Zero')
    libor_garch_fitted = garch_libor.fit()

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
        VaR = (port_ret - stats.norm.ppf(alpha) * port_vol) * value_port * -1
    elif(dist == 'student-t'):
        VaR = (port_ret - stats.t.ppf(alpha, DoF) * port_vol) * value_port * -1
    return(VaR)

CCC(df, start = "2011-03-02", stop = "2019-03-01", alpha = 0.975, dist = 'normal', DoF = 0)


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