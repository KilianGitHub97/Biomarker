# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:57:06 2021

@author: kilia
"""
#----- setup -----#
import os
import pandas as pd

#set working directory
wd = "C:\\Users\\kilia\\Documents\\GitHub\\Biomarker\\code"
os.chdir(wd)

#read in data
X = pd.read_csv("..\\data\\Biomarker.csv")

#options
pd.set_option("display.max_columns", 65)
pd.set_option("display.max_rows", 65)

#----- exploration -----#

'''
print(X.head(10))
print(X.shape)
print(X.dtypes)
print(X.columns)
print(X.describe())
print(X.isna().sum())
print(X.isnull().sum())
'''

#----- rename columns -----#

#extract colnames as list
col_names_original = list(X.columns)

#### delete unwanted character-combos from the column-names ####

#list with unwanted character-combos
trashs = ["(-)", "(ms)", "(-/min)", "(dB)", "(‰/min)", "(-/min2)",
          "(mg/day)", "-", ":", "(years)"]

#new list for the manipulated variable-names
new_cols = [] 

#assign original column names to new list for the upcoming loop
cols = col_names_original

#delete all unwanted character-combos
for trash in trashs:
    
    #for all unwanted symbols (trash), replace them with nothing        
    cols = [col.replace(trash,"") for col in cols]    

#assign new column names to the DataFrame        
X.columns = cols 

#### more variable-names cleaning ####

#strip white space from beginning and end
X.columns = X.columns.str.strip() 

#make everything lowercase
X.columns = X.columns.str.lower() 

#replace double spacing (2x since some variables have 3 consecutive whitespaces)
X.columns = X.columns.str.replace("  ", " ")
X.columns = X.columns.str.replace("  ", " ")

#simplify strings that start with numbers 
for i in range(X.shape[1]): 
    
    #get first four characters from string
    first_four = X.columns[i][:4] 
    
    #if there is a digit in the fist four characters of the colname, then delete the first four characters of the string
    if any(map(str.isdigit, first_four)): 
        X.rename(columns={X.columns[i]: X.columns[i][4:]}, inplace=True)  

#replace .1 with indication for monologue
X.columns = X.columns.str.replace(".1", "monologue")

#manually fixing the name that did not convert 
X.columns.values[58] = "gaping inbetween voiced intervals monologue"

#add indication for reading task column
for i in range(41,53):
    X.columns.values[i] = X.columns.values[i] + " reading"

#convert whitespace to underscore
X.columns = X.columns.str.replace(" ", "_")  

#save transformed columnnames
col_names_new = list(X.columns)

#create dictionary as xlsx and save it to folder: "report"
var_summary = {
    "var_nr" : [i for i in range(1, X.shape[1]+1)],
    "col_names_original" : col_names_original,
    "col_names_new" : col_names_new
    }
var_summary = pd.DataFrame(var_summary)

'''
var_summary.to_excel("..\\accompanying_docs\\dictionary.xlsx",
                     index=False,
                     header=True)
'''

#delete unneeded variables
del [col_names_new,
     col_names_original,
     cols,
     first_four,
     new_cols,
     trash,
     trashs]

#---- tidy values of columns ----#

#tidy anti..._therapy & benzo..._medication cols, which contain "Yes" + medication
for i in range(len(X)):
    if X.at[i, "antidepressant_therapy"] != "No":
        X.at[i, "antidepressant_therapy"] = "Yes"
    if X.at[i, "benzodiazepine_medication"] != "No":
        X.at[i, "benzodiazepine_medication"] = "Yes"

#check
print(X["benzodiazepine_medication"].value_counts())
print(X["antidepressant_therapy"].value_counts())

#---- Output ----#
X.to_csv("..\\data\\biomarker_clean.csv",
         header = True,
         index = False)

#clear environment
del [i,
     var_summary,
     wd,
     X]