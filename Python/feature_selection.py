#! /usr/bin/python

#importing system
import sys
# numpy and pandas for data manipulation
import pandas as pd
import numpy as np

# cross-validation
from sklearn.model_selection import cross_val_score

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# visualizations
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.base import clone 

# utilities
from itertools import combinations

# deactivating warning
pd.options.mode.chained_assignment = None
pd.options.mode.chained_assignment = None

# garbage collector
import gc

import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# udfs ----

# function for creating a feature selection dataframe
def fs_df(column_names,accuracy_original, accuracy_derived, delta):
    df = pd.DataFrame({'feature': column_names,
                       'accuracy_original': accuracy_original,
                        'accuracy_derived': accuracy_derived,
                        'accuracy_delta': delta}) \
           .sort_values('accuracy_delta', ascending = False) \
           .reset_index(drop = True)
    return df

### Start Feature Genetarion ###
def automate_add(df):
    return pd.DataFrame({c1 + '_sum_' + c2: df[c1] + df[c2] 
                  for c1, c2 in combinations(df.columns, 2)})
    
def automate_diff(df):
    return pd.DataFrame({c1 + '_diff_' + c2: df[c1] - df[c2] 
                  for c1, c2 in combinations(df.columns, 2)})

def automate_mult(df):
    return pd.DataFrame({c1 + '_mult_' + c2: df[c1].mul(df[c2], fill_value=0).replace([np.inf, np.NINF, np.nan], 0)
                      for c1, c2 in combinations(df.columns, 2)})
    
def automate_ratio(df):
    return pd.DataFrame({c1 + '_div_' + c2: df[c1].div(df[c2], fill_value=0).replace([np.inf, np.NINF, np.nan], 0)
                  for c1, c2 in combinations(df.columns, 2)})
    

def featureGeneration(df, operators):
    add = df.iloc[:,0:0]
    diff = df.iloc[:,0:0]
    mult = df.iloc[:,0:0]
    ratio = df.iloc[:,0:0]
    for i, operator in enumerate(operators):
        if operator == "sum":
            add = automate_add(df)
        elif operator == "diff":
            diff = automate_diff(df)
        elif operator == "mult":
            mult = automate_mult(df)
        elif operator == "ratio":
            ratio = automate_ratio(df)
    return pd.concat([add, diff, mult, ratio], axis=1, sort=False)


def extractDerivedFeatures(df_original, df_extended):
    defaultFeaturesList = list(df_original.columns.values)
    defaultFeaturesList.remove('class')

    
    return df_extended.drop(columns=defaultFeaturesList)

# Performance of each column plus the original dataset
def add_col_feat_imp(model, X, y, X_derived):
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # cross validation
    scores = cross_val_score(model_clone, X, y, cv=10)
    benchmark_score = scores.mean()
    # list for storing feature importances
    delta = []
    accuracy_derived = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_derived.columns:
        # clone the model to have the exact same specification as the one initially trained
        model_clone = clone(model)
        # cross validation
        cross_col_scores = cross_val_score(model_clone, pd.concat([X, X_derived[col]], axis=1), y, cv=10)
        col_score = cross_col_scores.mean()
        
        #Saving and reseting
        accuracy_derived.append(col_score)
        delta.append(col_score - benchmark_score)
        cross_col_scores = 0
        col_score = 0
        
        
    importances_df = fs_df(X_derived.columns, benchmark_score, accuracy_derived, delta)
    return importances_df

def feature_performance(df_original, df_extended):
    #Dividing labels and features
    y = np.array(df_original['class'])
    X = df_original.drop('class', axis = 1)

    # CROSS VALIDATION
    clf= RandomForestClassifier(n_estimators=100, random_state=20)

    # FEATURE GENERATION
    # operators = ["sum", "diff", "mult", "ratio"]
    # X_derived = featureGeneration(df_extended, operators)

    X_derived = extractDerivedFeatures(df_original, df_extended.drop('class', axis = 1))

    #Performance of each feature for the data set
    return add_col_feat_imp(clf, X, y, X_derived)

    



def main():
    pathFiles = pd.read_csv('./main_fs.csv')

    for i, column in pathFiles.iterrows():
        print("#### Processing", column[0])
        logging.warning(column[0])

        df_original = pd.read_csv(column[1])
        df_extended = pd.read_csv(column[2])
        if not 'class' in df_original.columns:
            print("There is no 'class' column on that database!")
            return

        importances_df = feature_performance(df_original, df_extended)
        importances_df.to_csv('./data/filter_selection/'+ column[0])
        gc.collect() 



if __name__ == "__main__":
    main()