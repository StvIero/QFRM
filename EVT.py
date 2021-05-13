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

##Appears that Credit-Suisse has the fattest tails. Next, plot qq-plot to confirm.

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
    # axs[subplot_row, subplot_col].set_title(name)
    # axs[subplot_row, subplot_col].legend()
    # axs[subplot_row, subplot_col].set_ylim(0, 2000)
    # axs[subplot_row, subplot_col].set_xlim(-0.4, 0.4)
plt.show()