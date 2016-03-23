import argparse
import numpy as np 
from sklearn.svm import LinearSVC
import sklearn.linear_model
from sklearn.metrics import accuracy_score
import pandas as pd 
import re
import math
from sklearn import preprocessing
from csv import DictReader, DictWriter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import sklearn
import csv
from sklearn.preprocessing import Imputer
from numpy import unravel_index
from sklearn.base import TransformerMixin
from sklearn.cross_validation import train_test_split


class DataFrameImputer(TransformerMixin):

	def __init__(self):
		"""Impute missing values.

		Columns of dtype object are imputed with the most frequent value 
		in column.

		Columns of other types are imputed with mean of column.

		"""
	def fit(self, X, y=None):

		self.fill = pd.Series([X[c].mean()
			if X[c].dtype == np.dtype('float64') or X[c].dtype == np.dtype('int64') else X[c].value_counts().index[0] for c in X],
			index=X.columns)

		return self


	def transform(self, X, y=None):
		return X.fillna(self.fill)

	def fit_transform(self,X,y=None):
		return self.fit(X,y).transform(X)




def End ():
	print "Question Finish"



if __name__ == "__main__":
	print('The scikit-learn version is {}.'.format(sklearn.__version__))
	parser = argparse.ArgumentParser(description='Question_1')

	args = parser.parse_args()

	train = list(csv.reader(open("train.data", 'r'), delimiter='\t'))

	test = list(csv.reader(open("test.data", 'r'), delimiter='\t'))

	laEn = preprocessing.LabelEncoder()

	lengthTrain = len(train) - 1
	lengthTest = len(test) - 1

	print lengthTrain
	print lengthTest

	threshold = 0.5*lengthTrain

	train_labels = list(csv.reader(open("Labels.csv", 'r'), delimiter='\t'))

	labels = pd.DataFrame(train_labels)

	
	labels.columns = labels.values[0]

	
	labels = labels[1:].reset_index(drop = True)

	

	intialtrain_df  = pd.DataFrame(train)

	intialtest_df = pd.DataFrame(test)


	intialtrain_df.columns = intialtrain_df.values[0]

	intialtest_df.columns = intialtest_df.values[0]

	features = intialtrain_df.columns

	train_df = intialtrain_df[1:].reset_index(drop=True)

	test_df = intialtest_df[1:].reset_index(drop=True)

	backUptrain_idf = train_df
	

	numerictraindf = train_df[train_df.columns[0:190]].apply(pd.to_numeric, errors='coerce')

	nonnumerictraindf = train_df[train_df.columns[190:230]]

	numerictestdf = test_df[test_df.columns[0:190]].apply(pd.to_numeric, errors='coerce')

	nonnumerictestdf = test_df[test_df.columns[190:230]]



	nonnumerictraindf = nonnumerictraindf.replace("", np.nan, regex=True)

	nonnumerictestdf = nonnumerictestdf.replace("", np.nan, regex=True)

	

	featurenum = 1
	
	extractedFeatures = []
	numericfeatures = []
	stringfeatures = []
	droppedfeatures = []
	for eachvalue in features:
		countval = 0

		if featurenum < 191:
			for each in numerictraindf[eachvalue]:
				if math.isnan(float(each)):
					countval = countval + 1
		else:
			for each in nonnumerictraindf[eachvalue]:
				if nonnumerictraindf[eachvalue].dtype == np.dtype('object'):
					if each is np.nan:
						countval = countval + 1
				else:
					countval = countval + 1


		if countval <= threshold:
			extractedFeatures.append(eachvalue)
			if featurenum < 191:
				numericfeatures.append(eachvalue)
			else:
				stringfeatures.append(eachvalue)
		else:
			droppedfeatures.append(eachvalue)


		featurenum = featurenum + 1
		
	print len(extractedFeatures)

	

	reduceNumericdf = numerictraindf[numericfeatures]
	reduceStringdf = nonnumerictraindf[stringfeatures]

	reduceNumericdf_Test = numerictestdf[numericfeatures]
	reduceStringdf_Test = nonnumerictestdf[stringfeatures]


	newtraindf = pd.concat([reduceNumericdf, reduceStringdf], axis=1)

	newTestdf = pd.concat([reduceNumericdf_Test, reduceStringdf_Test], axis=1)

	
	finaltrain_df = DataFrameImputer().fit_transform(newtraindf)
	finaltest_df = DataFrameImputer().fit_transform(newTestdf)


	count = 1

	EncodeTrainData = []

	for each in stringfeatures:
		if count == 1:
			EncodeTrainData = finaltrain_df[each]
			count = 2
		else:
			EncodeTrainData = np.column_stack((EncodeTrainData,finaltrain_df[each]))


	count = 1

	EncodeTestData = []

	for each in stringfeatures:
		if count == 1:
			EncodeTestData = finaltest_df[each]
			count = 2
		else:
			EncodeTestData = np.column_stack((EncodeTestData,finaltest_df[each]))


	EncodeData = np.vstack((EncodeTrainData,EncodeTestData))


	Encoded_X = np.reshape(EncodeData,(1, 2*lengthTrain*len(stringfeatures)))
	print Encoded_X.shape
	
	fullLabel = laEn.fit_transform(np.reshape(Encoded_X,(2*lengthTrain*len(stringfeatures), 1)))
	ShapedLabel = np.reshape(fullLabel,(1, 2*lengthTrain*len(stringfeatures)))

	print ShapedLabel.shape

	divideLabelEncode = np.reshape(ShapedLabel, (2, lengthTrain*len(stringfeatures)))

	print divideLabelEncode.shape
	print len(laEn.classes_)
	print divideLabelEncode[0].shape
	print divideLabelEncode[1].shape

	

	Encodedtrain_X = np.reshape(np.reshape(divideLabelEncode[0],(1, lengthTrain*len(stringfeatures))), (lengthTrain,len(stringfeatures))) 
	Encodedtest_X = np.reshape(np.reshape(divideLabelEncode[1],(1, lengthTest*len(stringfeatures))), (lengthTest,len(stringfeatures)))

	print Encodedtrain_X.shape
	print Encodedtest_X.shape

	
	count = 1
	X_TrainData = []

	for each in numericfeatures:
		if count == 1:
			X_TrainData = finaltrain_df[each]
			count = 2
		else:
			X_TrainData = np.column_stack((X_TrainData,finaltrain_df[each]))

	Full_X_TrainData = np.column_stack((X_TrainData,Encodedtrain_X))

	print Full_X_TrainData.shape

	count = 1
	X_TestData = []

	for each in numericfeatures:
		if count == 1:
			X_TestData = finaltest_df[each]
			count = 2
		else:
			X_TestData = np.column_stack((X_TestData,finaltest_df[each]))

	Full_X_TestData = np.column_stack((X_TestData,Encodedtest_X))

	print Full_X_TestData.shape


	X_dummytrain, X_dummytest, y_dummytrain, y_dummytest = train_test_split(Full_X_TrainData, labels, test_size=0.2, random_state=42)


	print X_dummytrain.shape
	print X_dummytest.shape
	print y_dummytrain.shape
	print y_dummytest.shape




	#SVM
	#ExtraTrees
	#Adaboost
	#Random
	#DecisionTree
	#GradientBoosting

	rng = np.random.RandomState(1)

	classifiers = [
	LinearSVC(C=0.01, penalty="l1", dual=False),
	ensemble.ExtraTreesClassifier(n_estimators = 100, random_state = 0),
	DecisionTreeClassifier(max_depth=10),
	ensemble.RandomForestClassifier(max_depth=10, n_estimators=100),
	ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=300),
	ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.1, max_depth = 10 , random_state = 0)]

	scores = np.zeros((6, 6))


	i = 0 

	for clf in classifiers:
		lsvc = clf.fit(X_dummytrain,y_dummytrain)
		model = SelectFromModel(lsvc, prefit = True)

		Train_new = model.transform(X_dummytrain)
		print Train_new.shape
		newindices = model.get_support(True)

		FinalTrainLessFeature = X_dummytrain[np.ix_(np.arange(40000), newindices)]
		FinalTestLessFeature = X_dummytest[np.ix_(np.arange(10000), newindices)]

		print FinalTrainLessFeature.shape
		print FinalTestLessFeature.shape
	

		j =0

		for clf in classifiers:
			rng = np.random.RandomState(1)

			estimate = clf.fit(FinalTrainLessFeature,y_dummytrain)

			predictions = estimate.predict(FinalTestLessFeature)

			scores[i][j] = accuracy_score(y_dummytest,predictions) 
			print scores

			j = j + 1

		FinalTestLessFeature = []
		FinalTrainLessFeature = []
		i = i + 1

	
	



	print scores

	i , j = unravel_index(scores.argmax(), scores.shape)


	lsvc = classifiers[i].fit(Full_X_TrainData,labels)
	
	model = SelectFromModel(lsvc, prefit = True)

	Train_new = model.transform(Full_X_TrainData)
	print Train_new.shape
	newindices = model.get_support(True)

	FinalTrainLessFeature = Full_X_TrainData[np.ix_(np.arange(lengthTrain), newindices)]
	FinalTestLessFeature = Full_X_TestData[np.ix_(np.arange(lengthTest), newindices)]

	print FinalTrainLessFeature.shape
	print FinalTestLessFeature.shape

	rng = np.random.RandomState(1)

	finalestimate = classifiers[j].fit(FinalTrainLessFeature,labels)

	predictions = finalestimate.predict(FinalTestLessFeature)
	

	print "In writePredictions"
	o = DictWriter(open("predictions1tocheck.csv", 'w'),["target"])
	for y_val in predictions:
		o.writerow({'target': y_val})

	End()
	



