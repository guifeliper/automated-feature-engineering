#! /usr/bin/python
# -*- coding: utf-8 -*-


# Metadata file is the the output data set
# Main file is the file with the path for all data sets
# This script generate features for one data set and evaluate the original and the extended

#importing system
import sys
import time
import os
# numpy and pandas for data manipulation
import pandas as pd
import numpy as np

# cross-validation
from sklearn.model_selection import cross_val_score

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# utilities
from itertools import combinations

# deactivating warning
pd.options.mode.chained_assignment = None

# garbage collector
import gc


### Start Feature Genetarion ###
def automate_add(df, add):
    add = pd.DataFrame({c1 + '_sum_' + c2: df[c1] + df[c2] 
                  for c1, c2 in combinations(df.columns, 2)})
    return add

def automate_diff(df, diff):
    diff = pd.DataFrame({c1 + '_diff_' + c2: df[c1] - df[c2] 
                  for c1, c2 in combinations(df.columns, 2)})
    return diff

def automate_mult(df, mult):
    mult = pd.DataFrame({c1 + '_mult_' + c2: df[c1].mul(df[c2], fill_value=0).replace([np.inf, np.NINF, np.nan], 0)
                    for c1, c2 in combinations(df.columns, 2)})
    return mult

def automate_ratio(df, ratio):
    ratio = pd.DataFrame({c1 + '_div_' + c2: df[c1].div(df[c2], fill_value=0).replace([np.inf, np.NINF, np.nan], 0)
                  for c1, c2 in combinations(df.columns, 2)})
    return ratio

def featureGeneration(df, operators):
    add = df.iloc[:,0:0]
    diff = df.iloc[:,0:0]
    mult = df.iloc[:,0:0]
    ratio = df.iloc[:,0:0]
    for i, operator in enumerate(operators):
        if operator == "sum":
            add = automate_add(df, add)
        elif operator == "diff":
            diff = automate_diff(df, diff)
        elif operator == "mult":
            mult = automate_mult(df, mult)
        elif operator == "ratio":
            ratio = automate_ratio(df, ratio)
    return pd.concat([add, diff, mult, ratio], axis=1, sort=False)


def rfClassifier(df):
# PREPARING MODELING
    # Labels are the values we want to predict
    labels = np.array(df['class'])
    # Remove the labels from the features
    # axis 1 refers to the columns
    features= df.drop('class', axis = 1)
    # Saving feature names for later use
    feature_list = list(features.columns)
    # Convert to numpy array
    features = np.array(features)

# CROSS VALIDATION
    clf= RandomForestClassifier(n_estimators=100, random_state=20)

    # 10-Fold Cross validation
    scores = cross_val_score(clf, features, labels, cv=10)

# METRICS
    # Model Accuracy, how often is the classifier correct?
    accuracy = round(scores.mean() * 100, 2)
    confidence_interval = round(scores.std(),4)
    # print("Accuracy:", round(accuracy, 2), '%.')
    return (accuracy, confidence_interval)


def drop_correlated(stFeatures):
    # Create correlation matrix
    corr_matrix = stFeatures.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # Drop features 
    stFeatures = stFeatures.drop(pd.Series(to_drop), axis=1)

    return stFeatures

def crisp(rawCSVdata, fileName, isDropCorrelated):
#PREPARE THE DATA
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = rawCSVdata.select_dtypes(include=numerics).drop('class', axis=1, errors='ignore').replace([np.inf, np.nan, np.NINF], 0)


# FEATURE GENERATION
    operators = ["sum", "diff", "mult", "ratio"]
    newFeatures = featureGeneration(df, operators)

    stFeatures = pd.concat([df, newFeatures], axis=1, sort=False)

    df_extended_no_correlated = drop_correlated(stFeatures)
    df_base_no_correlated = drop_correlated(df)

    df_new_features = pd.concat([df_extended_no_correlated, rawCSVdata['class']], axis=1, sort=False)
    df_base_numeric = pd.concat([df_base_no_correlated, rawCSVdata['class']], axis=1, sort=False)

    fac, fci = rfClassifier(df_new_features)
    bac, bci = rfClassifier(df_base_numeric)

    metaKnowledge = {
    'extended.accuracy':fac, 
    'base.accuracy':bac,
    'delta.accuracy': np.around(np.subtract(fac, bac),2),
    'ratio.accuracy': fac/bac if bac != 0 else 0,
    'extended.confidence.interval': fci,
    'base.confidence.interval': bci,
    'class': 1 if fac > bac  else 0
    }
    metaKnowledgeRow = pd.Series(metaKnowledge)

    df_new_features.to_csv('./data/extended/' + fileName + '.csv', index=False) 
    df_base_numeric.to_csv('./data/base/' + fileName + '.csv', index=False) 

    return metaKnowledgeRow



def main():
    pathFiles = pd.read_csv('main.csv')
    metaData = pd.read_csv('metadata.csv', index_col=0)
    start_time = time.time()
    print(os.getcwd())

    for i, pathFile in pathFiles.iterrows():
        print("#### Processing", pathFile[0])

        rawCSVdata = pd.read_csv(pathFile[1])
        if not 'class' in rawCSVdata.columns:
            print("There is no ´class´ column on that database!")
            return 
            
        metaData.loc[pathFile[0]] = crisp(rawCSVdata, pathFile[0], False)
        metaData.to_csv('metadata.csv')
        gc.collect() 

    print("===========")
    print("--- %s seconds ---" % (time.time() - start_time))
  

if __name__ == "__main__":
    main()