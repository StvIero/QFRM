###EVT Code###

#Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import seaborn as sns
from seaborn_qqplot import pplot
import pingouin as pg
import matplotlib.dates as mdates
from scipy.stats import genpareto

#Import data, set date column as index.
df = pd.read_csv('/Users/connorstevens/Documents/GitHub/qfrm_code/Data3/data_main.csv', index_col= 1)

#Loop through assets in portfolio and plot distributions to test for fat tails.

#Extract list of return names to plot.
assets = df.columns[df.columns.get_loc('ASML.AS_ret'): -1]
print(assets)

#Loop through assets and plot histogram of returns with same y-limit, x-limit and same number of bins for consistency.
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10,12), constrained_layout = True)
for count, name in enumerate(assets):
    subplot_col = 0
    subplot_row = count
    if count > 4:
        subplot_col = 1
        subplot_row -= 5
    axs[subplot_row, subplot_col].hist(df[name].dropna(), bins=20, label=name)
    axs[subplot_row, subplot_col].set_title(name)
    axs[subplot_row, subplot_col].legend()
    axs[subplot_row, subplot_col].set_ylim(0, 2000)
    axs[subplot_row, subplot_col].set_xlim(-0.4, 0.4)
plt.show()

#Plot qq-plots for normal and student-t distributions.
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10,12), constrained_layout = True)
for count, name in enumerate(assets):
    subplot_col = 0
    subplot_row = count
    if count > 4:
        subplot_col = 1
        subplot_row -= 5
    pg.qqplot(df[name].dropna(), dist = 'norm', ax = axs[subplot_row, subplot_col], confidence=False)
    axs[subplot_row, subplot_col].set_title(name)

plt.show()

##Based on qq-plot and kurtosis, we use Square Enix for fattest tails.

#Make losses positive, do not show gains.
losses = df['SQNXF_ret'].dropna()*-1
    #df['SQNXF_ret'][df.SQNXF_ret < 0].dropna()* -1

##Plot losses.
# Create figure and plot space
fig1, ax = plt.subplots(2)

# Add x-axis and y-axis
ax[0].plot(pd.to_datetime(losses.index.values), losses, label = 'SQNXF losses')

# Set title and labels for axes
ax[0].set(xlabel="Date",
       ylabel="Losses(%)")

# Define the date format
date_form = mdates.DateFormatter("%Y")
ax[0].xaxis.set_major_formatter(date_form)

# Ensure a major tick for each week using (interval=1)
ax[0].xaxis.set_major_locator(mdates.YearLocator())
fig1.legend()

#Plot histogram of losses alongside time series.
ax[1].hist(losses)
ax[1].set(ylabel='Frequency', xlabel = 'Losses(%)')
fig1.tight_layout()
fig1.show()

#Set excess loss threshold at 5%
threshold = 0.07
excess_losses = losses[losses > threshold]

plt.hist(excess_losses)
plt.show()


params = genpareto.fit(excess_losses)
print(genpareto.fit(excess_losses))

stats.t.fit(df['SQNXF_ret'].dropna())

#Plot theoretical distribution.

#Generate values for fitted Pareto distribution.
pareto_dist = stats.genpareto.rvs(params[0],params[1],params[2],size=len(excess_losses))

#Plot histogram of excess losses and theoretical Pareto distribution.
plt.hist(pareto_dist, label = 'Theoretical Pareto Distribution', alpha = 0.5)
plt.hist(excess_losses, alpha = 0.5, label = 'Excess Losses')
plt.ylabel('Frequency')
plt.xlabel('Losses(%)')
plt.legend()
plt.show()

#Plot CDF of excess losses and fitted Pareto distribution.
theoretical_cdf = stats.genpareto.cdf
x = np.linspace(threshold,0.22,1000)
plt.plot(x, theoretical_cdf(x, params[0],params[1],params[2]), label = 'Fitted Pareto CDF')
plt.scatter(excess_losses, theoretical_cdf(excess_losses, params[0],params[1],params[2]), label = 'Excess Losses CDF', color='red')
plt.xlabel('Losses(%)')
plt.ylabel('Cumulative Density')
plt.legend()
plt.show()

#QQ-Plot of excess losses and Pareto distribution.
sm.qqplot(excess_losses, stats.genpareto, fit=True, line="45")
plt.xlabel('GPD Theoretical Quantiles')
plt.ylabel('Excess Losses Quantiles')
plt.show()

#Kolmogorov-Smirnov test for fit of CDF.
ks_result = stats.kstest(excess_losses, genpareto.cdf, (params[0],params[1],params[2]))
print('p-value: ' + str(ks_result[1]) + ", therefore we cannot reject the null that the two samples are drawn from the same distribution.")

#Modify values to calculate VaR.
VaR_dist = (len(excess_losses)/len(losses)) * 1-pareto_dist.cdf
print(VaR_dist)

#Calculate VaR.
VaR975EVT = 0.07 + params[1]/params[2]*((len(losses)/len(excess_losses)*(1-0.25))**(-params[2]) -1)
print(VaR975EVT)
VaR99EVT = 0.07 + params[1]/params[2]*((len(losses)/len(excess_losses)*(1-0.1))**(-params[2]) -1)
print(VaR99EVT)

#Calculate ES.
ES975EVT = np.mean(VaR_dist[VaR_dist > VaR975EVT])
print(ES975EVT)
ES99EVT = np.mean(VaR_dist[VaR_dist > VaR99EVT])
print(ES99EVT)

sqnxf = df['SQNXF_ret'].dropna()*-1

#Calculate VaR by historical simulation.
VaR975HS = np.quantile(sqnxf, 0.975)
print(VaR975HS)
VaR99HS = np.quantile(sqnxf, 0.99)
print(VaR99HS)

VaR_test = 0.07 + params[1]/params[2]*((len(losses)/len(excess_losses)*(1-0.25))**(-params[2]) -1)
print(VaR_test)

VaR975EVT * (1/(1-params[2]) + (params[1] - params[2] * 0.07)/((1-params[2]) * VaR975EVT))