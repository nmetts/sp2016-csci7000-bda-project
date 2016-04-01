'''
Created on Mar 5, 2016

@author: Nicolas Metts
'''

import argparse
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from unbalanced_dataset.unbalanced_dataset import UnbalancedDataset

from unbalanced_dataset.over_sampling import OverSampler
from unbalanced_dataset.over_sampling import SMOTE
from unbalanced_dataset.pipeline import SMOTEENN
from unbalanced_dataset.pipeline import SMOTETomek
from unbalanced_dataset.under_sampling import UnderSampler
from sklearn.metrics.classification import accuracy_score
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

# Constants for classifier names
LOG_REG = 'log_reg'
SVM = 'svm'
ADA_BOOST = 'ada_boost'
GRADIENT_BOOST = 'gradient_boost'
RF = 'random_forest'
EXTRA_TREES = 'extra_trees'
BAGGING = 'bagging'

# Constants for sampling techniques
SMOTE_REG = "smote"
SMOTE_SVM = "smote_svm"
SMOTE_BORDERLINE_1 = "smote_borderline_1"
SMOTE_BORDERLINE_2 = "smote_borderline_2"
SMOTE_ENN = "smote_enn"
SMOTE_TOMEK = "smote_tomek"
UNDERSAMPLER = "undersampler"

class ClassifyArgs(object):
    """
    A class to represent the arguments used in classify
    """
    def __init__(self, data_file="../Data/orange_small_train.data",
                 train_file="../Data/orange_small_train.data",
                 test_file="../Data/orange_small_train.data",classify=False,
                 classifiers=None, kernel='rbf', cross_validate=False,
                 write_to_log=False, features=None, scale=False,
                 kfold=False, write_predictions=False):
        self.data_file = data_file
        self.train_file = train_file
        self.test_file = test_file
        self.classify = classify
        if classifiers is None:
            classifiers = ['svm']
        else:
            self.classifiers = classifiers
        self.kernel = kernel
        self.cross_validate = cross_validate
        self.write_to_log = write_to_log
        if features is None:
            self.features = []
        else:
            self.features = features
        self.scale = scale
        self.kfold = kfold
        self.write_predictions = write_predictions
    
    def __repr__(self):
        str_list = [self.data_file, self.train_file, self.test_file,
                    self.classify, self.kernel, self.cross_validate, self.scale,
                    self.kfold]
        str_list += self.features
        return "_".join([str(x) for x in str_list])

def write_log(out_file_name, args, classifier, precision, recall,
              true_count, actual_count, X_train, X_test, auc, accuracy):
    """
    Function to write results of a run to a file.
    """
    # Get the kernel type if classifier is SVM, otherwise just put NA
    get_kernel = lambda x: x == 'svm' and args.kernel or "NA"

    # Log important info
    log = [args.data_file, args.train_file, args.test_file,
           classifier, get_kernel(classifier), args.scale, len(X_train),
           len(X_test), precision, recall, accuracy, true_count, actual_count,
           auc, args.sampling_technique, args.sampling_ratio, args.select_best]
    with open(out_file_name, 'a') as f:
        out_writer = csv.writer(f, lineterminator='\n')
        out_writer.writerow(log)

def svm_classify(train_X, train_Y, test_X, test_Y, kernel, reg):
    """
    A function to run an SVM classification

    Args:
        train_X: training feature values
        train_Y: training labels
        test_X: testing feature values
        test_Y: testing labels
        kernel: a string representing the kernel to use
        reg: a float representing the regularization parameter
    """
    clf = SVC(kernel=kernel, C=reg)
    clf.fit(train_X, train_Y)
    sc = clf.score(test_X, test_Y)
    print('SVM score', kernel, reg, sc)

    return clf

def __print_and_log_results(clf, classifier, x_train, x_test, y_test, out_file_name,
                            args):
    predictions = clf.predict(x_test)
    precision = precision_score(y_test, predictions, [-1, 1])
    recall = recall_score(y_test, predictions, [-1, 1])
    auc_score = roc_auc_score(y_test, predictions, None)
    accuracy = accuracy_score(y_test, predictions)
    print "Train/test set sizes: " + str(len(x_train)) + "/" + str(len(x_test))
    print "Precision is: " + str(precision)
    print "Recall is: " + str(recall)
    print "AUC ROC Score is: " + str(auc_score)
    print "Accuracy is: " + str(accuracy)
    true_count = len([1 for p in predictions if p == 1])
    actual_count = len([1 for y in y_test if y == 1])
    print "True count (prediction/actual): " + str(true_count) + "/" + str(actual_count)

    if args.write_to_log:
    # Write out results as a table to log file
        write_log(out_file_name=out_file_name, args=args, classifier=classifier,
                    precision=precision, recall=recall,
                    true_count=true_count, actual_count=actual_count,
                    X_train=x_train, X_test=x_test,
                    auc=auc_score, accuracy=accuracy)

def __get_sample_transformed_examples(sample_type, train_x, train_y, ratio):
    sampler = None
    verbose = True
    if sample_type == SMOTE_REG:
        sampler = SMOTE(kind='regular', verbose=verbose, ratio=ratio, k=15)
    elif sample_type == SMOTE_SVM:
        # TODO: Make this configurable?
        svm_args = {'class_weight' : 'balanced'}
        sampler = SMOTE(kind='svm', ratio=ratio, verbose=verbose, k=15, **svm_args)
    elif sample_type == SMOTE_BORDERLINE_1:
        sampler = SMOTE(kind='borderline1', ratio=ratio, verbose=verbose)
    elif sample_type == SMOTE_BORDERLINE_2:
        sampler = SMOTE(kind='borderline2', ratio=ratio, verbose=verbose)
    elif sample_type == SMOTE_ENN:
        sampler = SMOTEENN(ratio=ratio, verbose=verbose, k=15)
    elif sample_type == SMOTE_TOMEK:
        sampler = SMOTETomek(ratio=ratio,verbose=verbose, k=15)
    elif sample_type == UNDERSAMPLER:
        sampler = UnderSampler(ratio=ratio, verbose=verbose, replacement=False,
                               random_state=17)
    else:
        print "Unrecoqnized sample technique: " + sample_type
        print "Returning original data"
        return train_x, train_y
    return sampler.fit_transform(train_x, train_y)

def __get_classifier_model(classifier, args):
    """
    Convenience function for obtaining a classification model

    Args:
        classifier(str): A string indicating the name of the classifier
        args: An arguments object

    Returns:
        A classification model based on the given classifier string
    """
    # Make SGD Logistic Regression model the default
    model = SGDClassifier(loss='log', penalty='l2', shuffle=True, n_iter=5)
    if classifier == LOG_REG:
        model = SGDClassifier(loss='log', penalty='l2', shuffle=True, n_iter=5,
                              random_state=179, n_jobs=-1)
    elif classifier == SVM:
        model = SVC(kernel=args.kernel, class_weight="balanced")
    elif classifier == ADA_BOOST:
        model = AdaBoostClassifier(n_estimators=300, random_state=17)
    elif classifier == RF:
        # Configure the classifier to use all available CPU cores 
        model = RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                       n_estimators=300, random_state=17)
    elif classifier == GRADIENT_BOOST:
        model = GradientBoostingClassifier(random_state=17, n_estimators=300)
    elif classifier == EXTRA_TREES:
        model = ExtraTreesClassifier(random_state=17, n_estimators=300, n_jobs=-1)
    elif classifier == BAGGING:
        model = BaggingClassifier(n_estimators=300, random_state=17, n_jobs=-1)
    return model


def main(args):
    out_file_name = "results.log"

    if args.classify:
        # Cast to list to keep it all in memory
        train = list(csv.reader(open(args.train_file, 'r')))
        test = list(csv.reader(open(args.test_file, 'r')))

        x_train = np.array(train[1:], dtype=float)
        
        x_test = np.array(test[1:], dtype=float)
        
        train_labels_file = open(args.train_labels)
        y_train = np.array([int(x.strip()) for x in train_labels_file.readlines()])

        test_labels_file = open(args.test_labels)
        y_test = np.array([int(x.strip()) for x in test_labels_file.readlines()])
        train_labels_file.close()
        test_labels_file.close()

        if args.sampling_technique:
            print "Attempting to use sampling technique: " + args.sampling_technique
            if args.sampling_ratio == float('NaN'):
                print "Unable to use sampling technique. Ratio is NaN."
            else:
                x_train, y_train = __get_sample_transformed_examples(args.sampling_technique,
                                                                     x_train, y_train,
                                                                     args.sampling_ratio)

        if args.scale:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.fit_transform(x_test)
        for classifier in args.classifiers:
            model = __get_classifier_model(classifier, args)
            print "Using classifier " + classifier
            print "Fitting data to model"
            clf = model.fit(x_train, y_train)
            if args.select_best:
                # Unable to use BaggingClassifier with SelectFromModel
                if classifier != BAGGING:
                    print "Selecting best features"
                    sfm = SelectFromModel(clf, prefit=True)
                    x_train = sfm.transform(x_train)
                    x_test = sfm.transform(x_test)
                    clf = model.fit(x_train, y_train)
            __print_and_log_results(clf, classifier, x_train, x_test, y_test,
                                    out_file_name, args)

    elif args.cross_validate:
        # Cast to list to keep it all in memory
        labels_file = open(args.labels)
        labels = np.array([int(x.strip()) for x in labels_file.readlines()])
        labels_file.close()
        data_file = open(args.data_file, 'r')
        data = list(csv.reader(data_file))
        data_file.close()
        examples = np.array(data[1:], dtype=float)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(examples, labels, test_size=0.2)

        if args.sampling_technique:
            print "Attempting to use sampling technique: " + args.sampling_technique
            if args.sampling_ratio == float('NaN'):
                print "Unable to use sampling technique. Ratio is NaN."
            else:
                X_train, y_train = __get_sample_transformed_examples(args.sampling_technique,
                                                                     X_train, y_train,
                                                                     args.sampling_ratio)
        if args.scale:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        for classifier in args.classifiers:
            print "Using classifier " + classifier
            model = __get_classifier_model(classifier, args)
            print "Fitting model"
            clf = model.fit(X_train, y_train)
            if args.select_best:
                if classifier != BAGGING:
                    sfm = SelectFromModel(clf, prefit = True)
                    X_train = sfm.transform(X_train)
                    X_test = sfm.transform(X_test)
                    clf = model.fit(X_train, y_train)
            print "Evaluating results"
            __print_and_log_results(clf, classifier, X_train, X_test, y_test,
                                    out_file_name, args)
    elif args.kfold:
        # Cast to list to keep it all in memory
        data_file = open((args.data_file, 'r'))
        data = list(csv.reader(data_file))
        data_file.close()
        labels_file = open(args.labels)
        labels = np.array([int(x.strip()) for x in labels_file.readlines()])
        labels_file.close()
        X = np.array(data, dtype=float)
        kf = KFold(len(X), n_folds=10, shuffle=True, random_state=42)
        for train, test in kf:
            print "kfold loop iterate"
            X_train, X_test, y_train, y_test = X[train], X[test], labels[train], labels[test]

            if args.sampling_technique:
                print "Attempting to use sampling technique: " + args.sampling_technique
                if args.sampling_ratio == float('NaN'):
                    print "Unable to use sampling technique. Ratio is NaN."
                else:
                    X_train, y_train = __get_sample_transformed_examples(args.sampling_technique,
                                                                     X_train, y_train,
                                                                     args.sampling_ratio)
            if args.scale:
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            for classifier in args.classifiers:
                print "Using classifier " + classifier
                model = __get_classifier_model(classifier, args)
                print "Fitting model"
                clf = model.fit(X_train, y_train)
                if args.select_best:
                    if classifier != BAGGING:
                        sfm = SelectFromModel(clf, prefit = True)
                        X_train = sfm.transform(X_train)
                        X_test = sfm.transform(X_test)
                        clf = model.fit(X_train, y_train)
                print "Evaluating results"
                __print_and_log_results(clf, classifier, X_train, X_test, y_test,
                                        out_file_name, args)
        print "kfold loop done"

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_file", help="Name of data file",
                           type=str, default="../Data/orange_small_train.data",
                           required=False)
    argparser.add_argument("--train_file", help="Name of train file",
                           type=str, default="../Data/orange_small_train.data",
                           required=False)
    argparser.add_argument("--test_file", help="Name of test file",
                           type=str, default="../Data/orange_small_test.data",
                           required=False)
    argparser.add_argument("--labels", help="Name of labels file",
                           type=str, default="../Data/orange_small_train_churn.labels",
                           required=False)
    argparser.add_argument("--test_labels", help="Name of test labels file",
                           type=str, default="../Data/orange_small_train_churn.labels",
                           required=False)
    argparser.add_argument("--train_labels", help="Name of train labels file",
                           type=str, default="../Data/orange_small_train_churn.labels",
                           required=False)
    argparser.add_argument("--classify", help="Classify using training and test set",
                           action="store_true")
    argparser.add_argument("--classifiers", help="A list of classifiers to use",
                           nargs='+', required=False, default=['svm'])
    argparser.add_argument("--kernel",
                           help="The kernel to be used for SVM classification",
                           type=str, default='rbf')
    argparser.add_argument("--cross_validate",
                           help="Cross validate using training and test set",
                           action="store_true")
    argparser.add_argument("--kfold", help="10-fold cross validation",
                           action="store_true")
    argparser.add_argument("--write_to_log", help="Send output to log file",
                           action="store_true")
    argparser.add_argument("--scale", help="Scale the data with StandardScale",
                           action="store_true")
    argparser.add_argument("--sampling_technique",
                          help="The sampling technique to use", type=str, required=False)
    argparser.add_argument("--sampling_ratio",
                          help="The sampling ratio to use", type=float,
                          default=float('NaN'), required=False)
    argparser.add_argument("--select_best", help="Select best features",
                           action="store_true")
    args = argparser.parse_args()
    main(args)
