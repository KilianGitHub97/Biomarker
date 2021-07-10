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
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.cluster import KMeans

#----- setup -----#

#set working directory
wd = "C:\\Users\\kilia\\Documents\\GitHub\\Biomarker\\code"
os.chdir(wd)

#read in data
X = pd.read_csv("..\\data\\biomarker_clean.csv")

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

#---------------------- plot mean differences ----------------------------#
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
#-------------------------------------------------------------------------#                                               
#delete unneeded variables
del [i,
     park_no,
     park_yes,
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
# sns.pairplot(X_pred,
#              hue="parkinson",
#              palette="Set2",
#              diag_kind="kde")

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

#--------------------- plot classification problem ---------------------#
#reduce data onto two axis
pca = PCA(n_components=2)
pca.fit(x_train)
PCs = pca.transform(x_train)

#how much information will be covered by the plot? approx. 39%
exp_var = pca.explained_variance_ratio_.sum().round(2)

#plot classification problem
pca_plot = sns.scatterplot(x = PCs[:,0],
                           y = PCs[:,1],
                           hue = y_train,
                           legend = True)
pca_plot.set(xlabel="Principal Component 1",
             ylabel="Principal Component 2",
             title="Classificationproblem in two principal components \n NOTE: the two PCs only cover approx. {} of the total variance!".format(exp_var))
#----------------------------------------------------------------------#

#LeaveOneOut Cross Validation
rkf_validation = RepeatedKFold(n_splits = 10, n_repeats = 5)

#Randomized Grid
models_hyperparameters = {
    "logistic_regression" : {
        "model" : LogisticRegression(solver="liblinear"),
        "hyperparameters" : {
            "C" : [0.5, 0.7, 1, 3, 5, 10]
            }
        },
    "quadratic_discriminant_analysis" : {
        "model" : QuadraticDiscriminantAnalysis(),
        "hyperparameters" : {
            "reg_param": list(np.arange(0,1,0.1))
            }
         },
    "support_vector_machine" : {
        "model" : svm.SVC(gamma = "auto", shrinking =True),
        "hyperparameters" : {
            "kernel" : ["linear", "poly", "rbf"],
            "C" : [0.1, 0.5, 1, 1.5, 2, 5, 10],
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

#recreate SVM-model
#get best parameters
par_supvec = results.iat[2,2]

supvec = svm.SVC(gamma = "auto",
                 shrinking = True,
                 C = par_supvec.get("C"),
                 degree = par_supvec.get("degree"),
                 kernel = par_supvec.get("kernel"))

#fit model
supvec.fit(x_train, y_train)

#mean and sd on 10 fits in-sample
val = cross_val_score(supvec, x_train, y_train, cv = 10)
val.mean()
val.std()

#in-sample performance
y_pred_supvec_train = pd.Series(supvec.predict(x_train))
confusion_matrix(y_train, y_pred_supvec_train)
print(classification_report(y_pred_supvec_train, y_train))

#out-of-sample-performance
y_pred_supvec_test = pd.Series(supvec.predict(x_test))
confusion_matrix(y_test, y_pred_supvec_test)
print(classification_report(y_pred_supvec_test, y_test))

#recreate logreg model
#get best parameters
par_logreg = results.iat[0,2]

logreg = LogisticRegression(C = par_logreg.get("C"))

#fit model
logreg.fit(x_train, y_train)

#mean and sd on 10 fits in-sample
val = cross_val_score(logreg, x_train, y_train, cv = 10)
val.mean()
val.std()

#in-sample performance
y_pred_logreg_train = pd.Series(logreg.predict(x_train))
confusion_matrix(y_train, y_pred_logreg_train)
print(classification_report(y_pred_logreg_train, y_train))

#out-of-sample-performance
y_pred_logreg_test = pd.Series(logreg.predict(x_test))
confusion_matrix(y_test, y_pred_logreg_test)
print(classification_report(y_pred_logreg_test, y_test))

#recreate QDA model
#get best parameters
par_qda = results.iat[1,2]

qda = QuadraticDiscriminantAnalysis(reg_param = par_qda.get("reg_param"))

#fit model
qda.fit(x_train, y_train)

#mean and sd on 10 fits in-sample
val = cross_val_score(qda, x_train, y_train, cv = 10)
val.mean()
val.std()

#in-sample performance
y_pred_qda_train = pd.Series(qda.predict(x_train))
confusion_matrix(y_train, y_pred_qda_train)
print(classification_report(y_pred_qda_train, y_train))

#out-of-sample-performance
y_pred_qda_test = pd.Series(qda.predict(x_test))
confusion_matrix(y_test, y_pred_qda_test)
print(classification_report(y_pred_qda_test, y_test))

#delete unneeded variables
del [model, 
     model_name,
     models_hyperparameters,
     mp,
     pca,
     pca_plot,
     PCs,
     exp_var,
     par_supvec,
     par_logreg,
     par_qda,
     y_pred_logreg_test,
     y_pred_logreg_train,
     y_pred_qda_test,
     y_pred_qda_train,
     y_pred_supvec_test,
     y_pred_supvec_train,
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