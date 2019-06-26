#! /usr/bin/python

#importing system
import sys
import os
# numpy and pandas for data manipulation
import pandas as pd
import numpy as np

import logging

logging.basicConfig(filename='drop.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def drop_high_correlated_feature(df):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    # Drop features 
    return df.drop(pd.Series(to_drop), axis=1)


def main():
    data_path = "./data/extended"
    filter_path = "./data/filter_extended"

    for filename in os.listdir(data_path):
        print("#### Processing ", filename)
        logging.warning(filename)
        csv = pd.read_csv(data_path + '/' + filename)
        csv = drop_high_correlated_feature(csv)
        csv.to_csv(filter_path + '/' + filename, index = False)

if __name__ == "__main__":
    main()