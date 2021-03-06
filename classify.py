'''
Created on Mar 5, 2016

@author: Nicolas Metts
'''

import argparse
import csv

from sklearn import  grid_search
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics.classification import accuracy_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree.tree import DecisionTreeClassifier
from unbalanced_dataset.over_sampling import SMOTE
from unbalanced_dataset.pipeline import SMOTEENN
from unbalanced_dataset.pipeline import SMOTETomek
from unbalanced_dataset.under_sampling import UnderSampler, ClusterCentroids, NearMiss, TomekLinks

from adasyn import ADASYN
import numpy as np


# Constants for classifier names
LOG_REG = 'log_reg'
SVM = 'svm'
ADA_BOOST = 'ada_boost'
GRADIENT_BOOST = 'gradient_boost'
RF = 'random_forest'
EXTRA_TREES = 'extra_trees'
BAGGING = 'bagging'
PASSIVE_AGGRESSIVE = 'passive_aggressive'
PERCEPTRON = 'perceptron'

# Constants for sampling techniques
SMOTE_REG = "smote"
SMOTE_SVM = "smote_svm"
SMOTE_BORDERLINE_1 = "smote_borderline_1"
SMOTE_BORDERLINE_2 = "smote_borderline_2"
SMOTE_ENN = "smote_enn"
SMOTE_TOMEK = "smote_tomek"
UNDERSAMPLER = "undersampler"
TOMEK_LINKS = "tomlek_links"
CLUSTER_CENTROIDS = "cluster_centroids"
NEARMISS = "near_miss"
ADASYN_SAMPLER = "adasyn"

class ClassifyArgs(object):
    """
    A class to represent the arguments used in classify
    """
    def __init__(self, data_file="../Data/orange_small_train.data",
                 train_file="../Data/orange_small_train.data",
                 test_file="../Data/orange_small_train.data",classify=False,
                 classifiers=None, kernel='rbf', cross_validate=False,
                 write_to_log=False, features=None, scale=False,
                 kfold=False, write_predictions=False, grid_search=False):
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
              true_count, actual_count, X_train, X_test, auc, accuracy,
              probablistic_prediction, prediction_threshold):
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

def __print_and_log_results(clf, classifier, x_train, x_test, y_test, out_file_name,
                            args):
    probablistic_predictions = False
    if args.predict_proba:
        predict_proba_func = getattr(clf, "predict_proba", None)
        if predict_proba_func is not None:
            probablistic_predictions = True
            prob_predictions = clf.predict_proba(x_test)
            predictions = []
            pos_predictions = []
            for prediction in prob_predictions:
                pos_predictions.append(prediction[1])
                if prediction[1] > args.predict_threshold:
                    predictions.append(1)
                else:
                    predictions.append(-1)
            pos_predictions = np.array(pos_predictions)
            mean_confidence = np.mean(pos_predictions)
            max_confidence = max(pos_predictions)
            min_confidence = min(pos_predictions)
            print "Mean confidence: " + str(mean_confidence)
            print "Max confidence: " + str(max_confidence)
            print "Min confidence: " + str(min_confidence)
            predictions = np.array(predictions)
        else:
            predictions = clf.predict(x_test)
    else:
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
                    auc=auc_score, accuracy=accuracy,
                    probablistic_prediction=probablistic_predictions,
                    prediction_threshold=args.predict_threshold)

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
    elif sample_type == ADASYN_SAMPLER:
        sampler = ADASYN(k=15,imb_threshold=0.6, ratio=ratio)
    elif sample_type == TOMEK_LINKS:
        sampler = TomekLinks()
    elif sample_type == CLUSTER_CENTROIDS:
        sampler = ClusterCentroids(ratio=ratio)
    elif sample_type == NEARMISS:
        sampler = NearMiss(ratio=ratio)
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
    model = SGDClassifier(loss='log', penalty='l2', shuffle=True, n_iter=5,
                          n_jobs=-1, random_state=179)
    if classifier == SVM:
        model = SVC(kernel=args.kernel, class_weight="balanced", cache_size=8096,
                    random_state=17, probability=True)
    elif classifier == ADA_BOOST:
        dt = DecisionTreeClassifier(max_depth=15, criterion='gini',
                                    max_features='auto', class_weight='balanced',
                                    random_state=39)
        model = AdaBoostClassifier(base_estimator=dt, n_estimators=400, random_state=17)
    elif classifier == RF:
        # Configure the classifier to use all available CPU cores 
        model = RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                       n_estimators=400, random_state=17,
                                       max_features='auto', max_depth=15,
                                       criterion='gini')
    elif classifier == GRADIENT_BOOST:
        model = GradientBoostingClassifier(random_state=17, n_estimators=400,
                                           max_features='auto')
    elif classifier == EXTRA_TREES:
        model = ExtraTreesClassifier(random_state=17, n_estimators=400, n_jobs=-1,
                                     class_weight='balanced', max_depth=15,
                                     max_features='auto', criterion='gini')
    elif classifier == BAGGING:
        dt = DecisionTreeClassifier(max_depth=15, criterion='gini',
                                    max_features='auto', class_weight='balanced',
                                    random_state=39)
        model = BaggingClassifier(base_estimator=dt, n_estimators=400,
                                  random_state=17, n_jobs=-1, max_features=0.8,
                                  max_samples=0.8, bootstrap=False)
    elif classifier == PASSIVE_AGGRESSIVE:
        model  = PassiveAggressiveClassifier(n_iter=10, class_weight='balanced',
                                           n_jobs=-1, random_state=41)
    elif classifier == PERCEPTRON:
        model = Perceptron(n_jobs=-1, n_iter=10, penalty='l2',
                           class_weight='balanced', alpha=0.25)
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
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.fit_transform(x_test)
        for classifier in args.classifiers:
            model = __get_classifier_model(classifier, args)
            print "Using classifier " + classifier
            print "Fitting data to model"
            if args.grid_search:
                print "Applying parameter tuning to model"
                if classifier == LOG_REG:
                    parameters = {'loss':('log','hinge'), 'penalty':('l2', 'l1'), 'shuffle':[True], 'n_iter':[5], 'n_jobs':[-1], 'random_state':[179]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == SVM:
                    parameters = {'kernel':('rbf', 'poly'), 'cache_size':[8096], 'random_state':[17]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == ADA_BOOST:
                    parameters = {'n_estimators':[300], 'random_state':[13]}
                    model = grid_search.GridSearchCV(model, parameters, scoring=roc_auc_score, verbose=2)
                elif classifier == RF:
                    parameters = {'criterion':('gini', 'entropy'), 'n_jobs':[-1], 'n_estimators':[300], 'random_state':[17]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == GRADIENT_BOOST:
                    parameters = {'n_estimators':[300], 'random_state':[17]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == EXTRA_TREES:
                    parameters = {'n_estimators':[300], 'random_state':[17], 'n_jobs':[-1], 'criterion':('gini', 'entropy'), 'max_features':('log2', 40, 0.4), 'max_features':[40, 0.4], 'bootstrap':[True, False], 'bootstrap_features':[True, False]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == BAGGING:
                    parameters = {'n_estimators':[300], 'random_state':[17], 'max_samples': [.4, 30],'max_features':[40, 0.4], 'bootstrap':[True, False], 'bootstrap_features':[True, False], 'n_jobs':[-1]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                print "Best params: " + str(model.best_params_)
                    
            clf = model.fit(x_train, y_train)
            print "Parameters used in model:"
            #print clf.get_params(deep=False)
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
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(examples, labels, test_size=0.1)

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
            if args.grid_search:
                print "Applying parameter tuning to model"
                if classifier == LOG_REG:
                    parameters = {'loss':('log','hinge'), 'penalty':('l2', 'l1'), 'shuffle':[True], 'n_iter':[5], 'n_jobs':[-1], 'random_state':[179]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == SVM:
                    parameters = {'kernel':('rbf', 'poly'), 'cache_size':[8096], 'random_state':[17]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == ADA_BOOST:
                    parameters = {'n_estimators':[300], 'random_state':[13]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == RF:
                    parameters = {'criterion':('gini', 'entropy'), 'n_jobs':[-1], 'n_estimators':[300], 'random_state':[17]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == GRADIENT_BOOST:
                    parameters = {'n_estimators':[300], 'random_state':[17]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == EXTRA_TREES:
                    parameters = {'n_estimators':[300], 'random_state':[17], 'n_jobs':[-1], 'criterion':('gini', 'entropy'), 'max_features':('log2', 40, 0.4), 'max_features':[40, 0.4], 'bootstrap':[True, False], 'bootstrap_features':[True, False]}
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
                elif classifier == BAGGING:
                    #parameters = {'n_estimators' : [400], 'random_state' : [17],
                    #              'max_samples' : np.arange(0.5, 0.9, 0.1),
                    #              'max_features' : np.arange(0.5, 0.9, 0.1),
                    #              'bootstrap':[False], 'bootstrap_features':[False], 'n_jobs':[-1]}
                    parameters = {"base_estimator__criterion" : ["gini", "entropy"],
                                  "base_estimator__splitter" : ["best", "random"],
                                  "base_estimator__max_depth" : [10, 15, 20, 25], 
                                  "base_estimator__class_weight" : ['balanced'],
                                  "base_estimator__max_features" : ['auto', 'log2']
                                  }
                    model = grid_search.GridSearchCV(model, parameters, scoring='roc_auc', verbose=2)
            clf = model.fit(X_train, y_train)
            if args.grid_search:
                print "Best params: " + str(model.best_params_)
            if args.select_best:
                if classifier != BAGGING:
                    print "Selecting best features"
                    sfm = SelectFromModel(clf, prefit = True)
                    X_train = sfm.transform(X_train)
                    X_test = sfm.transform(X_test)
                    clf = model.fit(X_train, y_train)
            print "Evaluating results"
            __print_and_log_results(clf, classifier, X_train, X_test, y_test,
                                    out_file_name, args)
    elif args.kfold:
        # Cast to list to keep it all in memory
        data_file = open(args.data_file, 'r')
        data = list(csv.reader(data_file))
        data_file.close()
        labels_file = open(args.labels)
        labels = np.array([int(x.strip()) for x in labels_file.readlines()])
        labels_file.close()
        X = np.array(data[1:], dtype=float)
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
    # Data file should be used when the task is cross-validation or k-fold
    # validation
    argparser.add_argument("--data_file", help="Name of data file",
                           type=str, default="../Data/orange_small_train.data",
                           required=False)
    # Labels is intended to be used for the cross-validation or k-fold validation
    # task
    argparser.add_argument("--labels", help="Name of labels file",
                           type=str, default="../Data/orange_small_train_churn.labels",
                           required=False)
    # Train file and test file are intended for classification (the classify
    # option)
    argparser.add_argument("--train_file", help="Name of train file",
                           type=str, default="../Data/orange_small_train.data",
                           required=False)
    argparser.add_argument("--test_file", help="Name of test file",
                           type=str, default="../Data/orange_small_test.data",
                           required=False)
    # Test and train labels are needed for the classify task
    argparser.add_argument("--test_labels", help="Name of test labels file",
                           type=str, default="../Data/orange_small_train_churn.labels",
                           required=False)
    argparser.add_argument("--train_labels", help="Name of train labels file",
                           type=str, default="../Data/orange_small_train_churn.labels",
                           required=False)
    # The classify task uses pre-split train/test files with train/test labels
    argparser.add_argument("--classify", help="Classify using training and test set",
                           action="store_true")
    argparser.add_argument("--classifiers", help="A list of classifiers to use",
                           nargs='+', required=False, default=['log_reg'])
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
    argparser.add_argument("--grid_search", help="Use grid search",
                           action="store_true")
    argparser.add_argument("--select_best", help="Select best features",
                           action="store_true")
    argparser.add_argument("--predict_proba", help="Select best features",
                           action="store_true")
    argparser.add_argument("--predict_threshold",
                          help="The prediction threshold to use", type=float,
                          default=0.55, required=False)
    args = argparser.parse_args()
    main(args)
