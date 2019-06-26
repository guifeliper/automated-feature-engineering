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

# garbage collector
import gc
import time
import logging

logging.basicConfig(filename='app4.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# udfs ----

# function for creating a feature selection dataframe
def fs_df(column_names,accuracy_original, accuracy_derived, delta):
    df = pd.DataFrame({'feature': column_names,
                       'accuracy_original': accuracy_original,
                        'accuracy_derived': accuracy_derived,
                        'accuracy_delta': delta }) \
           .sort_values('accuracy_delta', ascending = False) \
           .reset_index(drop = True)
    
    df['class'] = df['accuracy_delta'].apply(lambda x: '1' if x > 0 else '0')
    return df



def extractDerivedFeatures(df_original, df_extended):
    defaultFeaturesList = list(df_original.columns.values)
    defaultFeaturesList.remove('class')

    
    return df_extended.drop(columns=defaultFeaturesList, axis = 1, errors = 'ignore')

# Performance of each column plus the original dataset
def drop_col_feat_imp(model, X, y, X_derived_only):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    # model_clone.random_state = random_state
    # training and scoring the benchmark model
    # model_clone.fit(X_train, y_train)
    # benchmark_score = model_clone.score(X_train, y_train)
    scores = cross_val_score(model_clone, X, y, cv=10)
    benchmark_score = scores.mean() * 100
    # list for storing feature importances
    delta = []
    accuracy_derived = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_derived_only.columns:
        model_clone = clone(model)
        
        # model_clone.random_state = random_state
        X_plus = X.drop(col, axis = 1)
        cross_col_scores = cross_val_score(model_clone, X_plus, y, cv=10)
        col_score = cross_col_scores.mean() * 100
        
        #Saving and reseting
        accuracy_derived.append(col_score)
        delta.append(col_score - benchmark_score)
        cross_col_scores = 0
        col_score = 0
        
    
    importances_df = fs_df(X_derived_only.columns, benchmark_score, accuracy_derived, delta)
    return importances_df
def drop_high_correlated_feature(df):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    # Drop features 
    return df.drop(pd.Series(to_drop), axis=1)


def feature_performance(df_original, df_extended):
    #Drop high correlated Features from derived features
    df = drop_high_correlated_feature(df_extended)
    
    #Dividing labels and features
    y = np.array(df['class'])
    X = df.drop('class', axis = 1)

    # CROSS VALIDATION
    clf= RandomForestClassifier(n_estimators=100, random_state=20, n_jobs=-1)

    #Keeping only the derived features
    X_derived_only = extractDerivedFeatures(df_original, X)

    #Performance of each feature for the data set
    return drop_col_feat_imp(clf, X, y, X_derived_only)

    



def main():
  
    pathFiles = pd.read_csv('./main_fs.csv')
    OutsideStart = time.time()
    for i, column in pathFiles.iterrows():
        
        start = time.time()
        print("#### Processing", column[0])
        logging.warning(column[0])

        df_original = pd.read_csv(column[1])
        df_extended = pd.read_csv(column[2])
        if not 'class' in df_original.columns:
            print("There is no 'class' column on that database!")
            return

        importances_df = feature_performance(df_original, df_extended)
        importances_df.to_csv('./data/filter_selection/'+ column[0])
        end = time.time()
        print("Process %s finished on %0.2f secs" % (column[0], (end - start)))
        logging.warning("Process %s finished on %0.2f secs" % (column[0], (end - start)))
        end = 0
        gc.collect() 
    OutsideEnd = time.time()
    print("Total time of %0.2f secs" % ((OutsideEnd - OutsideStart)))
    logging.warning("Total time of %0.2f secs" % ((OutsideEnd - OutsideStart)))



if __name__ == "__main__":
    main()