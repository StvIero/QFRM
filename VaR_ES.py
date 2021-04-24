#Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model

#Import loss dataframe.
df = pd.read_csv('/Users/connorstevens/Documents/GitHub/QFRM/Data/loss_df.csv', index_col= 0)

#Multiplied all returns by 100 due to errors form GARCH function.
#Fit AEX GARCH model.
garch_aex = arch_model(df['aex_ret'].dropna() * 100, vol='garch', p=1, o=0, q=1, dist='Normal', mean = 'Zero')
aex_garch_fitted = garch_aex.fit(update_freq=5)

#Fit Nikkei GARCH model.
garch_nikkei = arch_model(df['nikkei_ret'].dropna() * 100, vol='garch', p=1, o=0, q=1, dist='Normal', mean = 'Zero')
nikkei_garch_fitted = garch_nikkei.fit(update_freq=5)

#Fit JSE GARCH model.
garch_jse = arch_model(df['jse_ret'].dropna() * 100, vol='garch', p=1, o=0, q=1, dist = 'Normal', mean = 'Zero')
jse_garch_fitted = garch_jse.fit()

##Fit LIBOR GARCH model.
garch_libor = arch_model(df['libor'].dropna() * 100, vol='garch', p=1, o=0, q=1, mean='Zero')
libor_garch_fitted = garch_libor.fit()

#Forecast variance.
aex_var_for = aex_garch_fitted.forecast(horizon= 1, start = 0, reindex= False)

nikkei_var_for = nikkei_garch_fitted.forecast(horizon= 1, start = 0, reindex= False)

jse_var_for = jse_garch_fitted.forecast(horizon= 1, start = 0, reindex= False)

libor_var_for = libor_garch_fitted.forecast(horizon= 1, start = 0, reindex= False)

##Include variance forecasts into dataframe.
#Create columns to store variance in
df['aex_var'] = np.zeros(len(df['aex_ret']))
df['nikkei_var'] = np.zeros(len(df['aex_ret']))
df['jse_var'] = np.zeros(len(df['aex_ret']))
df['libor_var'] = np.zeros(len(df['aex_ret']))

#Loop through variance forecasts and pull variance forecasts out of lists.
for i in range(1, len(df['aex_var'])):

    df['aex_var'].iloc[i] = aex_var_for.variance.values[i - 1][0]
    df['nikkei_var'].iloc[i] = nikkei_var_for.variance.values[i - 1][0]
    df['jse_var'].iloc[i] = jse_var_for.variance.values[i - 1][0]
    df['libor_var'].iloc[i] = libor_var_for.variance.values[i - 1][0]



#Create correlation matrix which will be held constant.
df['port_var'] = np.zeros(len(df['aex_ret']))
port_corr = np.array(df[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor']].corr())
variance = np.array(df[['aex_var', 'nikkei_var', 'jse_var', 'libor_var']])
weights = np.array([0.6, 0.6, 0.3, -0.5])

for l in range(1, np.shape(df)[0]):
    #Create covariance matrix to fill.
    port_covar = np.zeros((4,4))
    aex_pos_val = (100_000_000 * 0.6) * np.exp(np.sum(df['aex_ret'].iloc[1:l + 1]))
    nikkei_pos_val = (100_000_000 * 0.6) * np.exp(np.sum(df['nikkei_ret'].iloc[1:l + 1]))
    jse_pos_val = (100_000_000 * 0.3) * np.exp(np.sum(df['aex_ret'].iloc[1:l + 1]))
    libor_pos_val = (100_000_000 * -0.5) * (np.exp(np.sum(df['libor_change'].iloc[1:l + 1]))) #+ (100_000_000 * -0.5) * df['libor'].iloc[l + 1] / 250
    portval = np.sum([aex_pos_val, nikkei_pos_val, jse_pos_val, libor_pos_val])
    weights = np.array([aex_pos_val/portval, nikkei_pos_val/portval, jse_pos_val/portval, libor_pos_val/portval])
    for j in range(0, 3):
        for i in range(0, 3):
            port_covar[i, j] = port_corr[i,j] * variance[l, i] * variance[l, j]

    #Calculate portfolio variance.
    df['port_var'].iloc[l] = np.dot(weights.T, np.dot(port_covar, weights))



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
