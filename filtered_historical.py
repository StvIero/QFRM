#Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats

def FHS(df, start, stop, alpha, sig_window, lamb):
