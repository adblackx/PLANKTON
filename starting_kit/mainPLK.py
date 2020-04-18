"""
Created on Sat Mar 27 2020
Last revised: April 4, 2020
@author: mouloua ramdane


"""
model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir); 

import pickle
import numpy as np   
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import plkClassifier as plkc

import warnings

with warnings.catch_warnings():

	import matplotlib.pyplot as plt
	import pandas as pd

	from libscores import get_metric
	import numpy as np
	import plkPreprocessing as prep

	from sklearn.metrics import make_scorer
	from sklearn.model_selection import cross_val_score

	from sklearn.neural_network import MLPClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.svm import SVC
	from sklearn.gaussian_process import GaussianProcessClassifier
	from sklearn.gaussian_process.kernels import RBF
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.metrics import balanced_accuracy_score as sklearn_metric

	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import RandomizedSearchCV
	from sklearn.ensemble import StackingClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.base import BaseEstimator
	from sklearn.datasets import load_wine
	from sklearn.datasets import load_iris
	from data_manager import DataManager


def testplkClassifier(X, Y, model_name, model_list):
	testAssist = plkc.plkAssitClassifier(model_name, model_list, X, Y)
	best_model_name, best_model_list = testAssist.compareModel()
	model_prefinal = testAssist.find_best_param_MODEL(best_model_name, best_model_list)
	#print("model_prefinal ", model_prefinal)

	"""for i in range(len(model_prefinal)):
		M = plkc.Classifier(X,Y)
		M.process(X,Y, model_process = model_list[i] )
		M.cross_validation_Classifier()
		M.training_score_Classifier()"""

	model_final = testAssist.voting(model_prefinal)
	print("DEBUT VOTING ")
	M1 = plkc.Classifier(X,Y)
	#print("model_final",model_final)
	M1.process(X,Y, model_process = model_final )
	A = M1.cross_validation_Classifier()
	print("CV VOTING: ", A.mean())
	print("metric VOTING: ", M1.training_score_Classifier() )



def testplkModel(X, Y):
	metric_name1, scoring_function1 = get_metric()

	A = plkm.plkClassifier()
	#A.fit(X_train, Y_train)
	"""	M = plkc.Classifier(A.xPLK,A.yPLK)
	M.testModel( model_process = A )
	scores = M.cross_validation_Classifier()"""
	res = cross_val_score(A, X, Y, cv=5 , scoring = make_scorer(scoring_function1))
	print("A cross_validation_Classifier:  ", res)

	B = plkm.plkClassifier()
	res = cross_val_score(B, X, Y, cv=5 , scoring = make_scorer(scoring_function1))
	print("B cross_validation_Classifier:  ", res)

if __name__=="__main__":

	model_name = ["Nearest Neighbors", "Random Forest"]

	model_list = [KNeighborsClassifier(1),  RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=19, min_samples_leaf= 7)]
	
	data_dir = 'public_data_raw_gaiasavers'          # POUR TRAVAILLER SUR RAW DATA
	data_name = 'plankton'
	D = DataManager(data_name, data_dir) 

	X_train = D.data['X_train']
	Y_train = D.data['Y_train'].ravel()

	print(len(X_train[0]))
	print(len(X_train))
	print(len(Y_train))

	preop = prep.Preprocessor()

	X_train1, Y_train1 = prep.Preprocessor.outliersDeletion(X_train,Y_train)

	X_train1 = preop.fit_transform(X_train1, Y_train1)

	print(len(X_train1[0]))
	print(len(X_train1))
	print(len(Y_train1))




	model_nameS = ["ExtraTreesClassifier", "RandomForestClassifier"]
	model_listS = [ ExtraTreesClassifier() ,RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1)]

	#testplkClassifier(X_train1, Y_train1, model_nameS, model_listS)
	a = plkc.findModel()
	m = a.getModel(X_train1, Y_train1)
	print(m)
'''
	#X_Random = np.random.rand(10752,203) #105752 lignes et 203 colonnes pour les images
	#Y_Random = np.random.randint(7,size=10752) #105752 lignes et 203 colonnes pour les images
	Data = load_wine()
	X_Random = Data.data
	Y_Random = Data.target


	testplkClassifier(X_Random, Y_Random, model_nameS, model_listS)'''
	#testplkModel(X_train, Y_train)
