#Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from arch import arch_model
from scipy import stats
initial_value = 100_000_000
#Import data.
data = pd.read_csv('/Users/connorstevens/Documents/GitHub/QFRM/Data/loss_df.csv', index_col=0)

df = data[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor', 'loss']].dropna().copy()
df.describe()
window = 250
var_es975FH = np.zeros((np.shape(df)[0] - window, 3))
var_es99FH = np.zeros((np.shape(df)[0] - window, 3))

def FHS(df, alpha, sig_window, lamb, VaRES):
    # alpha = 0.975
    # sig_window = 100
    # lamb = 0.94
    # start = "2011-03-02"
    # stop = "2021-03-01"
    #Use some data to calculate initial sigma.
    initial_sigma = np.square(np.std(df.iloc[0: sig_window + 1], axis = 0))

    #Create columns to store variance estimates.
    df[['aex_ewmavar', 'nikkei_ewmavar', 'jse_ewmavar', 'libor_ewmavar']] = 0

    #Calcualte an initial sigma for ewma and set the last day of initial_sigma estimation window equal to initial_sigma.
    df.iloc[sig_window + 1, [df.columns.get_loc("aex_ewmavar"), df.columns.get_loc("nikkei_ewmavar"), df.columns.get_loc("jse_ewmavar"), df.columns.get_loc("libor_ewmavar")]] = initial_sigma

    #Loop through dataframe and forecast variance.
    for j in range(4, 8):
        for i in range(sig_window + 2, len(df['aex_ewmavar'])):
            df.iloc[i, j] = lamb * df.iloc[i - 1, j] + (1 - lamb) * np.square(df.iloc[i - 1, j - 4])

    #Create sigma forecast vector.
    sig_forecast = np.zeros(4)
    sig_forecast = np.array(lamb * df[['aex_ewmavar', 'nikkei_ewmavar', 'jse_ewmavar', 'libor_ewmavar']].iloc[-1]) + np.array((1 - lamb) * np.square(df[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor']].iloc[-1]))

    #Plot vol.
    # Create figure and plot space
    fig1, ax = plt.subplots()

    # Add x-axis and y-axis
    ax.plot(pd.to_datetime(df.index.values), 100 * np.sqrt(df[['aex_ewmavar', 'nikkei_ewmavar', 'jse_ewmavar', 'libor_ewmavar']]), label = ['AEX variance', 'Nikkei variance', 'JSE variance', 'LIBOR variance'])

    # Set title and labels for axes
    ax.set(xlabel="Date",
           ylabel="EWMA Volatility Forecast(%)")

    # Define the date format
    date_form = mdates.DateFormatter("%Y")
    ax.xaxis.set_major_formatter(date_form)

    # Ensure a major tick for each week using (interval=1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig1.legend()
    fig1.show()

    #Calculate Zhat values.

    #Create dataframe to store simulated returns for VaR calculation.
    #z is innovations and x is returns.
    dfSim = pd.DataFrame(np.zeros((np.shape(df)[0], 9)))
        #np.zeros((df.index.get_loc(stop) - df.index.get_loc(start) + 1, 9)))
    dfSim.columns = ['aex_z', 'nikkei_z', 'jse_z', 'libor_z', 'aex_x', 'nikkei_x', 'jse_x', 'libor_x', 'port_x']

    #Calculate Zhat-values
    # dfSim.loc['aex_z'] = np.divide(df.aex_ret, df.aex_ewmavar)
    # dfSim.loc['nikkei_z'] = np.divide(df.nikkei_ret, df.nikkei_ewmavar)
    # dfSim.loc['jse_z'] = np.divide(df.jse_ret, df.jse_ewmavar)
    # dfSim.loc['libor_z'] = np.divide(df.libor, df.libor_ewmavar)
    for j in range(0, 4):
        for i in range(0, np.shape(dfSim)[0]):
            dfSim.iloc[i, j] = df.iloc[i, j]/ np.sqrt(df.iloc[i, j + 4])

    #Calculte Xhat values (returns based on extracted innovations and forecasted volatility).
    # dfSim.loc['aex_x'] = np.multiply(dfSim.aex_z, sig_forecast[0])
    # dfSim.loc['nikkei_x'] = np.multiply(dfSim.nikkei_z, sig_forecast[1])
    # dfSim.loc['jse_x'] = np.multiply(dfSim.jse_z, sig_forecast[2])
    # dfSim.loc['libor_x'] = np.multiply(dfSim.libor_z, sig_forecast[3])
    for j in range(0, 4):
        for i in range(0, np.shape(dfSim)[0]):
            dfSim.iloc[i , j + 4] = dfSim.iloc[i, j] *  np.sqrt(sig_forecast[j])

    #Calculate porfolio returns.
    for i in range(0, np.shape(dfSim)[0]):
        dfSim.iloc[i, -1] = np.sum(np.multiply(dfSim.iloc[i, 4:8], np.array([0.6, 0.6, 0.3, -0.5])))

    var_x = np.quantile(dfSim['port_x'].iloc[101:], alpha)
    var = initial_value * var_x
    es = np.mean(dfSim.iloc[101:,dfSim.columns.get_loc("port_x")][dfSim.iloc[101:,dfSim.columns.get_loc("port_x")]  * initial_value >= var]) * 100000000

    if VaRES == 'VaR':
        return (var)
    elif VaRES == 'ES':
        return (es)




    # plt.plot(100 * np.sqrt(df[['aex_ewmavar', 'nikkei_ewmavar', 'jse_ewmavar', 'libor_ewmavar']]), label = ['aex_var', 'nikkei_var', 'jse_var', 'libor_var'])
    # axes = plt.axes()
    # axes.set_xticks(["2011-08-05", "2012-08-06", "2013-08-06"])
    # plt.axes.set_xticklabels(['2011','2012','2013'])
    # plt.legend()
    # plt.show()

df = data[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor', 'loss']].dropna().copy()
print('97.5% VaR is ' + str(round(FHS(df, alpha=0.975, lamb=0.94, sig_window=100, VaRES = 'VaR'), 2)))

df = data[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor', 'loss']].dropna().copy()
print('97.5% ES is ' + str(round(FHS(df, alpha=0.975, lamb=0.94, sig_window=100, VaRES = 'ES'), 2)))

df = data[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor', 'loss']].dropna().copy()
print('99% VaR is ' + str(round(FHS(df, alpha=0.99, lamb=0.94, sig_window=100, VaRES = 'VaR'), 2)))

df = data[['aex_ret', 'nikkei_ret', 'jse_ret', 'libor', 'loss']].dropna().copy()
print('99% ES is ' + str(round(FHS(df, alpha=0.99, lamb=0.94, sig_window=100, VaRES = 'ES'), 2)))


####BACKTESTING
window = 1000
var_es975FH = np.zeros((np.shape(df)[0] - window, 3))
var_es99FH = np.zeros((np.shape(df)[0] - window, 3))
# es975 = np.zeros((np.shape(df)[0] - window, 3))
# es99 = np.zeros((np.shape(df)[0] - window, 3))


for i in range(1, (np.shape(df)[0] - window)): #2103):
    # var_es975FH[i, 0] = FHS(df.iloc[i: i + window, :], alpha=0.975, sig_window=100, lamb=0.94, VaRES= 'VaR')
    # var_es975FH[i, 1] = FHS(df.iloc[i: i + window, :], alpha=0.975, sig_window=100, lamb=0.94, VaRES= 'ES')
    # var_es975FH[i, 2] = df.iloc[i, -1]

    var_es99FH[i, 0] = FHS(df.iloc[i: i + window, :], alpha=0.99, sig_window=100, lamb=0.94, VaRES= 'VaR')
    var_es99FH[i, 1] = FHS(df.iloc[i: i + window, :], alpha=0.99, sig_window=100, lamb=0.94, VaRES= 'ES')
    var_es99FH[i, 2] = df.iloc[i, -1]


pd.DataFrame(var_es99FH).to_csv('/Users/connorstevens/Documents/GitHub/QFRM/Plots/var_es99FH.csv')

#plt.plot(var_es975[1:, 0], label = '97.5% VaR')
# plt.plot(var_es975[1: 1], alpha = 0.5, label = '97.5% ES')
# plt.plot(var_es975[1:, 2], alpha = 0.7, label = 'Returns')
# plt.legend()
pd.to_datetime(df.index.values)
# plt.show()
window_start_index = np.shape(df)[0] - window

index = pd.to_datetime(df.iloc[window + 1:, :].index.values)
plt.plot(index, var_es975FH[1:, 1], label = '97.5% ES')
plt.plot(index,var_es975FH[1:, 0], label = '97.5% VaR')
plt.plot(index,data['loss'].iloc[window+1:-1], alpha = 0.5, label = 'Returns')
plt.ylabel('Losses (Euros)')
plt.xlabel('Date')
plt.legend()
plt.show()


FHS(df, alpha=0.95, sig_window=100, lamb=0.94)
