'''
Created on Mar 5, 2016

@author: Nicolas Metts
'''
import csv
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer

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
    args = argparser.parse_args()
    file_type = 'small'
    if 'large' in args.data_file:
        file_type = 'large'
    categorical_index = 190
    if file_type == 'large':
        categorical_index = 14740
    data_file = open(args.data_file)
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
    imp = Imputer(missing_values='NaN', strategy=args.fill_in, axis=0)
    imp.fit(numerical_features)
    numerical_features_filled_in = imp.transform(numerical_features)
    print "Missing numerical values filled in"

    enc = OneHotEncoder()
    enc.fit(new_cat_features)
    categorical_transformed = enc.transform(new_cat_features).toarray()
    # Note: Using OneHotEncoder absolutely explodes the number of columns and
    # thus the data size. Will likely need to find a different approach.
    print "Categorical features encoded"

    print "Numerical shape is: " + str(new_cat_features.shape)
    print "Categorical shape is: " + str(categorical_transformed.shape)
    all_features = np.concatenate((new_cat_features, categorical_transformed), axis=1)
    num_features = all_features.shape[1]
    print "There are: " + str(num_features) + " features"

    header = ["Feature" + str(x) for x in range(num_features)]
    print "Creating file: " + args.processed_file
    processed_file = open(args.processed_file, 'w')
    writer = csv.writer(processed_file)
    writer.writerow(header)
    for feature in all_features:
        writer.writerow(feature)
    processed_file.close()
    print "Pre-Processed file completed"