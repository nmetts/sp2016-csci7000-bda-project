'''
Created on Mar 5, 2016

A module for preprocessing raw data files from the KDD Cup 2009 dataset.

@author: Nicolas Metts
'''
import csv
import numpy as np
import argparse
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer

def __sklearn_preprocess(data_file_name, fill_in, processed_file_name,
                         categorical_index):
    """
    Use sklearn preprocessing module to preprocess raw data file. This function
    fills in missing numerical values with the given fill_in strategy. In addition,
    categorical features are transformed to indices.

    Args:
        data_file_name(str): The path to the raw data file
        fill_in(str): The fill-in strategy to use
        processed_file_name(str): The name (including path) of the resulting processed file
        categorical_index(int): The index where categorical features begin

    """
    data_file = open(data_file_name)
    data = list(data_file.readlines())[1:]
    data_file.close()
    numerical_features = []
    categorical_features = []
    for line in data:
        features = line.split("\t")
        numerical_features.append([np.nan if x == '' else float(x) for x in features[0:categorical_index]])
        # How should we fill in missing categorical features?
        categorical_features.append(['Unknown' if x == '' else x for x in features[categorical_index:]])
    numerical_features = np.array(numerical_features)
    categorical_features = np.array(categorical_features)
    num_cat_features = categorical_features.shape[1]
    new_cat_features = []
    # Transform strings into numerical values by column
    for i in range(num_cat_features):
        le = LabelEncoder()
        col = categorical_features[:,i]
        le.fit(col)
        new_cat_features.append(list(le.transform(col)))
    new_cat_features = np.array(new_cat_features).transpose()
    imp = Imputer(missing_values='NaN', strategy=fill_in, axis=1)
    imp.fit(numerical_features)
    numerical_features_filled_in = imp.transform(numerical_features)
    print "Missing numerical values filled in"

    #enc = OneHotEncoder()
    #enc.fit(new_cat_features)
    #categorical_transformed = enc.transform(new_cat_features).toarray()
    # Note: Using OneHotEncoder absolutely explodes the number of columns and
    # thus the data size. Will likely need to find a different approach.
    print "Categorical features encoded"

    print "Numerical shape is: " + str(numerical_features_filled_in.shape)
    print "Categorical shape is: " + str(new_cat_features.shape)
    all_features = np.concatenate((numerical_features_filled_in, new_cat_features), axis=1)
    num_features = all_features.shape[1]
    print "There are: " + str(num_features) + " features"

    header = ["Feature" + str(x) for x in range(num_features)]
    dir_name = os.path.dirname(data_file_name)
    print "Creating file: " + dir_name + "/" + processed_file_name
    processed_file = open(dir_name + "/" + processed_file_name, 'w')
    writer = csv.writer(processed_file)
    writer.writerow(header)
    for feature in all_features:
        writer.writerow(feature)
    processed_file.close()
    print "Pre-Processed file completed"

def __pandas_preprocess(data_file_name, categorical_index, num_features,
                        processed_file_name):
    """
    A function to preprocess a file using Pandas. Columns with less than 10% of
    rows containing data are dropped, as are columns with a standard deviation
    of 0. Categorical features are transformed using a one hot approach, with a
    column for NA values.

    Args:
        data_file_name(str): The path to the raw data file
        categorical_inde(int): The index where categorical features begin
        num_features(int): The number of features in the data file
        processed_file_name(str): The name (including path) of the resulting processed file
    """
    data = pd.read_csv(data_file_name, sep="\t")
    numerical_columns = ["Var" + str(i) for i in range(1, categorical_index + 1, 1)]
    categorical_columns = ["Var" + str(i) for i in range(categorical_index, num_features + 1, 1)]
    remove = []
    count = data.count(axis=0)
    print "Removing extraneous columns"
    for col in data.columns:
        if col in numerical_columns:
            # Remove columns with constant values
            if data[col].std() == 0:
                remove.append(col)
        # Remove columns where less than 20% of rows contain data
        if count[col] < 20000:
            remove.append(col)
    remove = set(remove)
    data.drop(remove, axis=1, inplace=True)
  
    numerical_features = pd.DataFrame()
    for numerical_column in numerical_columns:
        if numerical_column in data:
            feature = data[numerical_column]
            print "Filling in missing values for: " + numerical_column  
            feature.fillna(data[numerical_column].mean(), inplace=True)
            numerical_features = pd.concat([numerical_features, feature], axis=1)
            data.drop(numerical_column, axis=1, inplace=True)
    cat_features = pd.DataFrame()
    print "Transforming categorical data"
    for column in categorical_columns:
        if column in data:
            print "Transforming column: " + column
            feature = data[column]
            counts = feature.value_counts()
            # Following procedure used by winning KDD Cup 2009 team and only
            # keeping the top 10 categorical features
            if len(counts) > 10:
                least_used_counts = feature.value_counts()[10:]
                least_used = [x[0] for x in least_used_counts.iteritems()]
                feature.replace(to_replace=least_used, value="other", inplace=True)
            feature_transformed = pd.get_dummies(feature, dummy_na=True,
                                                 prefix=column)
            cat_features = pd.concat([cat_features, feature_transformed], axis=1)
            data.drop(column, axis=1, inplace=True)
    data = pd.concat([numerical_features, cat_features], axis=1)
    print "Preprocessed DataFrame info: "
    print data.info()
    dir_name = os.path.dirname(data_file_name)
    print "Writing file: " + dir_name + "/" + processed_file_name
    data.to_csv(dir_name + "/" + processed_file_name, index=False)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_file", help="Name of data file",
                           type=str, default="../Data/orange_small_train.data",
                           required=True)
    argparser.add_argument("--processed_file", help="Name of processed file",
                           type=str, default="../Data/orange_small_train_proc.data",
                           required=True)
    argparser.add_argument("--fill-in",
                           choices = ["median", "mean", "most_frequent"])
    argparser.add_argument("--use_library", choices=["sklearn", "pandas"])
    args = argparser.parse_args()
    num_features = 230
    categorical_index = 190
    if 'large' in args.data_file:
        categorical_index = 14740
        num_features = 15000
    
    if args.use_library == "sklearn":
        __sklearn_preprocess(args.data_file, args.fill_in, args.processed_file, categorical_index)
    elif args.use_library == "pandas":
        __pandas_preprocess(args.data_file, categorical_index, num_features, args.processed_file)
