# -*- coding: utf-8 -*-
"""
This file will do PCA analysis and such
some FA too

Created on Tue May  4 11:50:54 2021

@author: MauritsOever
"""

# packages and set directory
import os
os.chdir(r"C:\Users\gebruiker\Documents\GitHub\QFRM")
from data_puller3000 import DataPuller_assignment3 # get dataloader function

df = DataPuller_assignment3() # get data