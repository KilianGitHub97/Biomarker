# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:43:06 2021

@author: kilia
"""

#----- libraries -----#
import os
import pandas as pd
import numpy as np
import seaborn as sns

#----- setup -----#

#set working directory
wd = "C:\\Users\\kilia\\Documents\\GitHub\\Biomarker\\code"
os.chdir(wd)

#read in data
X = pd.read_csv("..\\data\\biomarker_clean.csv")
X.columns
#options
pd.set_option("display.max_columns", 65)
pd.set_option("display.max_rows", 65)

#---- feature modification & engineering -----#

#make criterion column for early-onset parkinson disease
#for the sake of simplicity, pd and rbd patients were all labeled as ill

#make empty list for criterion
parkinson = []

#make values for parkinson (list)
for i in range(len(X)):
    
    #if the value of participant_code contains HC (index for control-group),
    #then assign No to parkinson.list
    if "HC" in X["participant_code"][i]:
        parkinson.append(0)
        
    #else assign No
    else: 
        parkinson.append(1)

#add parkinson.list to the DataFrame        
X["parkinson"] = parkinson

#make missing values NaN
X = X.replace('-', np.nan)

#make all data numerical (convert string values into dummies)
X["gender"] = X["gender"].replace({'F': 0, 'M': 1})

#list with columns that include strings with Yes/No
col_strs = ["positive_history_of_parkinson_disease_in_family",
            "antidepressant_therapy",
            "antiparkinsonian_medication",
            "antipsychotic_medication",
            "benzodiazepine_medication"]

#replace Yes with 1; and No with 0 for all columns of col_strs
for string in col_strs:
    X[string] = X[string].replace({"Yes" : 1, "No" : 0})

#delete features where all answers are No
print(X["antiparkinsonian_medication"].value_counts())
print(X["levodopa_equivalent"].value_counts())
print(X["antipsychotic_medication"].value_counts())

X = X.drop(columns=["antiparkinsonian_medication",
                    "levodopa_equivalent",
                    "antipsychotic_medication"])

del [wd,
     col_strs,
     parkinson,
     string]

#----- descriptive statistics -----#

#disease
X["parkinson"].value_counts().plot(kind="bar")
print(X["parkinson"].value_counts(normalize=True))

#hist of age by parkinson
X.hist("age",
       by="parkinson")

#gender
#count by parkinson
print(X.loc[X["parkinson"] == 1]["gender"].value_counts())
print(X.loc[X["parkinson"] == 0]["gender"].value_counts())

#barplot of gender
X["gender"].value_counts().plot(kind="bar")

#barplot of gender by parkinson
X.groupby(["gender", "parkinson"])["gender"].count().plot(kind="bar")

#create correlationmatrix & heatmap
cormat = X.loc[:, X.columns != 'participant_code'].corr()
sns.heatmap(cormat, cbar=True)

del cormat

#----- mean comparisons -----#

#check columns that unclude NaN and drop columns (axis=1 for columns)
X.isna().sum()
X_nona = X.dropna(axis=1)

#make two datasets separate datasets that only include diseased /control participants.
park_yes = X_nona.loc[X["parkinson"] == 1, X_nona.columns != "participant_code"]
park_no = X_nona.loc[X["parkinson"] == 0, X_nona.columns != "participant_code"]

#make dictionary for the comparison of the means of diseased vs. control group
#NOTE: some variables are of Binomial nature, the mean can be interpretted as a proportion
desc_stat = {
    "col_names" : list(park_yes.columns),
    "mean_yes" : list(park_yes.mean()),
    "sd_yes" : list(park_yes.std()),
    "mean_no" : list(park_no.mean()),
    "sd_no" : list(park_no.std()),
    "difference_in_mean" : list(park_no.mean() - park_yes.mean())
    }

#convert dictionary to dataframe
desc_stat = pd.DataFrame(desc_stat)

