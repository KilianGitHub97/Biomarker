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
#Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
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

#delete unneeded variables
del [i,
     park_no,
     park_yes]
#----- Predict early-onset parkinson disease -----#

#prepare data
X_pred = X_nona.drop(["age",
                      "gender",
                      "participant_code",
                      "antidepressant_therapy",
                      "benzodiazepine_medication",
                      "clonazepam"],
                      axis=1)

X_pred.columns
#split data into train/test by a 2/10 vs 8/10 ratio for model assessment
x_train, x_test, y_train, y_test = train_test_split(X_pred.drop("parkinson", axis=1),
                                                    X_pred["parkinson"],
                                                    test_size = 0.2)

#is the assessment data still balanced?
len(y_train)
len(y_test)
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)

#scale dataset
sc = StandardScaler()
x_train = sc.fit_transform(x_train, y_train)
x_test = sc.transform(x_test)

#10-Fold Cross Validation
rkf_validation = RepeatedKFold(n_splits=10, n_repeats=2)

#Randomized Grid
#LDA has a closed-form solution and therefore has no hyperparameters.
#The solution can be obtained using the empirical sample class covariance
#matrix. Shrinkage is used when there are not enough samples. In that case
#the empirical covariance matrix is often not a very good estimator.
models_hyperparameters = {
    "logistic_regression" : {
        "model" : LogisticRegression(solver="liblinear"),
        "hyperparameters" : {
            "penalty" : ["l1", "l2"],
            "C" : [0.5, 0.7, 1, 3, 5, 10]
            }
        },
    "linear_discriminant_analysis" : {
        "model" : LinearDiscriminantAnalysis(solver="lsqr"),
        "hyperparameters" : {
            "shrinkage": list(np.arange(0,1,0.1))
            }
        },
    "support_vector_machine" : {
        "model" : svm.SVC(gamma = "auto", shrinking = True),
        "hyperparameters" : {
            "kernel" : ["linear", "poly", "rbf", "sigmoid"],
            "C" : [0.1, 0.5, 1, 1.5, 2, 5, 10],
            "coef0" : [0.1, 0.5, 0.8, 1, 1.5, 2, 5],
            "degree" : [i for i in range(1,6)]
            }
        }
}

#results from hyperparameter tuning
results = []

#hyperparameter tuning
for model_name, mp in models_hyperparameters.items():
    model = GridSearchCV(mp["model"], 
                         mp["hyperparameters"],
                         cv=rkf_validation,
                         return_train_score=False)
    model.fit(x_train, y_train)
    results.append({
        "model": model_name,
        "best_score": model.best_score_,
        "best_params": model.best_params_
    })

results = pd.DataFrame(results, columns=['model','best_score','best_params'])

#a priori probability
X["parkinson"].value_counts(normalize=True)

#get best parameters
par = results.iat[2,2]

#recreate model
supvec = svm.SVC(gamma = "auto",
                 shrinking = True,
                 C = par.get("C"),
                 coef0 = par.get("coef0"),
                 degree = par.get("degree"),
                 kernel = par.get("kernel"))

supvec.fit(x_train, y_train)
supvec.score(x_test, y_test)

val = cross_val_score(supvec, x_train, y_train, cv = 10)
val.mean()
val.std()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
randfor = RandomForestClassifier()
randfor.fit(x_train,y_train)
randfor.score(x_test, y_test)

from sklearn.model_selection import cross_val_score
val = cross_val_score(randfor, x_train, y_train, cv = 10)
val.mean()
val.std()

gradb = GradientBoostingClassifier()
gradb.fit(x_train ,y_train)
gradb.score(x_test, y_test)
val = cross_val_score(gradb, x_train, y_train, cv = 10)
val.mean()
val.std()