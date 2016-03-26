'''

@author: Nicolas Metts
'''
import argparse
import csv
import numpy as np
import os
import smote

def restricted_float(x):
    x = float(x)
    if x <= 0.0 or x > 0.5:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 0.5]"%(x,))
    return x

def __create_train_and_test_files(data_file_name, test_perc, labels_file,
                                  file_prefix):
    """
    A function to create train and test files and their corresponding label files.

    Args:
        data_file_name(str): The name of a file containing examples to be split
        into train and test sets
        test_perc(float): The percentage of test examples
        labels_file(str): The name of a file containing labels for the examples
        file_prefix(str): The prefix for the newly created test, train, and labels files
    """
    dir_name = os.path.dirname(data_file_name)
    file_prefix = dir_name + "/" + file_prefix
    data_lines = []
    with open(data_file_name) as data_file:
        data_lines = list(csv.reader(data_file))
    label_file = open(labels_file)
    labels = [int(x.strip()) for x in label_file.readlines()]
    label_file.close()

    header = data_lines[0]
    examples = data_lines[1:]
    test_size = int(test_perc * len(examples))
    test_indices = np.random.choice(range(len(examples)), test_size, replace=False)
    test_examples = [examples[x] for x in test_indices]
    test_labels = [labels[x] for x in test_indices]
    train_examples = [x for ind, x in enumerate(examples) if ind not in test_indices]
    train_labels = [x for ind, x in enumerate(labels) if ind not in test_indices]
    # Write the train examples and labels
    print "Writing train file and train labels"
    with open(file_prefix + ".train", 'w') as train_file, \
        open(file_prefix + ".train.labels", 'w') as train_labels_file:
        train_writer = csv.writer(train_file)
        train_writer.writerow(header)
        for example in train_examples:
            train_writer.writerow(example)
        for label in train_labels:
            train_labels_file.write(str(label) + "\n")

    # Write the test examples
    print "Writing test file"
    with open(file_prefix + ".test", 'w') as test_file:
        test_writer = csv.writer(test_file)
        test_writer.writerow(header)
        for example in test_examples:
            test_writer.writerow(example)

    # Write the test labels
    print "Writing test labels"
    with open(file_prefix + ".test.labels", 'w') as test_labels_file:
        for label in test_labels:
            test_labels_file.write(str(label) + "\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_file", help="Name of data file",
                           type=str, default="../Data/orange_small_train.data",
                           required=False)
    argparser.add_argument("--labels_file", help="Name of test file",
                           type=str, default="../Data/orange_small_train_churn.labels",
                           required=False)
    argparser.add_argument("--test_perc", help="Percentage of test examples",
                           type=restricted_float, default=0.1, required=False)
    argparser.add_argument("--file_name", help="Name of data file",
                           type=str, required=True)
    args = argparser.parse_args()
    __create_train_and_test_files(args.data_file, args.test_perc,
                                  args.labels_file, args.file_name)
    