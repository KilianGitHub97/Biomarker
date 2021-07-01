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
trashs = ["(-)", ".1", "(ms)", "(-/min)", "(dB)", "(‰/min)", "(-/min2)",
          "(mg/day)", "-", ":", "(years)"]

#new list for the manipulated variable-names
new_cols = [] 

#delete all unwanted character-combos
for trash in trashs:
    
    #load in the original column names within the first iteration
    if trash == trashs[0]: 
        cols = col_names_original
        
    #update column names by deleting unwanted strings   
    cols = [col.replace(trash,"") for col in cols]
    
    #when the last iteration is completed -> save the modified list to the new_cols list
    if trash == trashs[len(trashs)-1]:
        new_cols = cols

#assign new column names to the DataFrame        
X.columns = new_cols 

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

#write dictionary file to report folder
'''
var_summary.to_excel("..\\report\\dictionary.xlsx",
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

#tidy anti..._therapy col, which contains "Yes" and medication
for i in range(len(X)):
    if X.at[i, "antidepressant_therapy"] != "No":
        X.at[i, "antidepressant_therapy"] = "Yes"

#check
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