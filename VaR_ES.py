#Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arch

#Import loss dataframe.
df_loss = pd.read_csv('/Users/connorstevens/Documents/GitHub/QFRM/Data/loss_df.csv', index_col= 0)

garch = arch_model(returns, vol='garch', p=1, o=0, q=1)
garch_fitted = garch.fit()

