# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:43:06 2021

@author: kilia
"""

#----- libraries -----#
import os
import pandas as pd
import matplotlib as mpl
import numpy as  np

#----- setup -----#

#set working directory
wd = "C:\\Users\\kilia\\Documents\\GitHub\\Biomarker\\code"
os.chdir(wd)

#read in data
X = pd.read_csv("..\\data\\biomarker_clean.csv")

#options
pd.set_option("display.max_columns", 65)
pd.set_option("display.max_rows", 65)

#---- descriptive statistics ----#

#hist of age

#feature engineering
