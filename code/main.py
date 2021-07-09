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
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.cluster import KMeans

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

#plot mean differences

#pivot DataFrame
desc_stat_plot = desc_stat.melt(id_vars="col_names",
                                value_vars=["mean_yes", "mean_no"],
                                value_name= "means")
 
#make lineplot with rotated axis                               
mean_plt = sns.lineplot(data = desc_stat_plot,
                        x = "col_names",
                        y = "means",
                        hue = "variable",
                        marker="o")
mean_plt.set_xticklabels(labels=list(desc_stat["col_names"]),
                         rotation=90)
mean_plt.set(xlabel="column name",
             ylabel="mean",
             title="Differences in mean between \n parkinson-patients (mean_yes) and control (mean_no)")
                                               
#delete unneeded variables
del [i,
     park_no,
     desc_stat_plot,
     mean_plt]

#----- Predict early-onset parkinson disease -----#

#prepare data
X_pred = X_nona.drop(["age",
                      "gender",
                      "participant_code",
                      "antidepressant_therapy",
                      "benzodiazepine_medication",
                      "clonazepam"],
                      axis=1)

#plot the whole dataset
sns.pairplot(X_pred,
             hue="parkinson",
             palette="Set2",
             diag_kind="kde")

# stratified split of data into train (80%) and test (20%) with sampling (shuffle)
#Test data will be used for model assessment
x_train, x_test, y_train, y_test = train_test_split(X_pred.drop("parkinson", axis=1),
                                                    X_pred["parkinson"],
                                                    test_size = 0.2,
                                                    shuffle=True,
                                                    stratify=X_pred["parkinson"])

#is the assessment data well balanced?
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
                          cv=rkf_validation, #use 2x repeated 10-fold cv
                          return_train_score=False)
    model.fit(x_train, y_train)
    results.append({
        "model": model_name,
        "best_score": model.best_score_,
        "best_params": model.best_params_
    })

#convert results to more readable DataFrame format
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

#out-of-sample performance
supvec.fit(x_train, y_train)
supvec.score(x_test, y_test)

#mean and sd on 10 fits in-sample
val = cross_val_score(supvec, x_train, y_train, cv = 10)
val.mean()
val.std()

#delete unneeded variables
del [model, 
     model_name,
     models_hyperparameters,
     mp,
     par,
     rkf_validation, 
     sc,
     val,
     x_train,
     x_test,
     y_train,
     y_test]

#----- Trying to recover the labels with KMeans -----#

#prepare data
X_kmeans = X_pred.drop("parkinson", axis=1)

sc = StandardScaler()
X_kmeans = sc.fit_transform(X_kmeans)

#modelling KMeans algorithm
kmeans = KMeans(n_clusters=2, n_init = 10)
kmeans.fit(X_kmeans)

#predict labels 
pred_labels = kmeans.fit_predict(X_kmeans)

#compare predicted labels with true labels
confusion_matrix(pred_labels, X_pred["parkinson"])
print(classification_report(pred_labels, X_pred["parkinson"]))

#delete unneeded variables
del [kmeans,
     pred_labels,
     sc,
     X_kmeans]

#----- predict parkinson state labels -----#

y = park_yes["overview_of_motor_examination_updrs_iii_total"]
