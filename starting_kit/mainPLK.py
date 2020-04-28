"""
Created on Sat Mar 27 2020
Last revised: April 18, 2020
@author: Mouloua Ramdane

Ajout de données Random, wine et iris pour les tests
Ajout de tests
testplkClassifier -> code avant la creation de finModel, deux code similaires
testplkModel -> tests le preprocessing s'il fonctionne, pour voir la diff avec et sans
RAW DATA
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
import os

import warnings

with warnings.catch_warnings():

	import matplotlib.pyplot as plt
	import pandas as pd

	from libscores import get_metric
	import numpy as np
	import plkPreprocessing as prep

	from sklearn.metrics import make_scorer
	from sklearn import metrics

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
	from model import model
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.pipeline import Pipeline
	import plkPreprocessing as prep




def testclassCVM(X, Y, model):
	print("testclassCVM : BEGIN")
	cvm = plkc.classCVM(X,Y)
	cvm.process(X,Y,model)
	cvm.cross_validation_Classifier()
	cvm.training_score_Classifier()
	print("testclassCVM : END")

def testAssistModel(X, Y, model_name, model_list):
	print("testAssistModel : BEGIN")
	a = plkc.assistModel(X,Y)
	a.setModels(model_name, model_list)
	model = a.getModel()
	model.fit(X,Y)
	print("testAssistModel : END")

def testModel(X, Y):
	print("testLoadModel : BEGIN")

	a = model(
	model_name = ["RandomForestClassifier"],
	model_list =[ RandomForestClassifier()] )

	a.fit(X,Y)
	#print(a.get_classes())
	#a.fit(X,Y)
	testclassCVM(X,Y,a)
	"""scoring_function1 = getattr(metrics, "balanced_accuracy_score")
	res = cross_val_score(a, X, Y, cv=5 , scoring = make_scorer(scoring_function1))
	print("cross_validation_Classifier:  ", res)
	print("cross_validation_Classifier (moyenne)  ", res.mean())	"""
	print("testLoadModel : END")

if __name__=="__main__":

	data_dir = 'public_data_raw_gaiasavers'          # POUR TRAVAILLER SUR RAW DATA
	data_name = 'plankton'
	D = DataManager(data_name, data_dir) 

	X_train = D.data['X_train']
	Y_train = D.data['Y_train'].ravel()

	print(len(X_train[0]))
	print(len(X_train))
	print(len(Y_train))

	
			

	
	#TEST WITH RANDOM DATA
	#X_train = np.random.rand(100,203) #105752 lignes et 203 colonnes pour les images
	#Y_train = np.random.randint(7,size=100) #105752 lignes et 203 colonnes pour les images
	#pipe_class = ExtraTreesClassifier() 
	#testclassCVM(X_train, Y_train, pipe_class)

	#model_nameS = ["ExtraTreesClassifier", "RandomForestClassifier"]
	#model_listS = [ ExtraTreesClassifier() ,RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1)]
	#testAssistModel(X_train, Y_train,model_nameS, model_listS)

	"""model_nameS = ["ExtraTreesClassifier", "RandomForestClassifier"]
	model_listS = [ ExtraTreesClassifier() ,RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1)]
	testAssistModel(X_train, Y_train,model_nameS, model_listS)"""
	testModel(X_train,Y_train)