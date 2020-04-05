"""
Created on Sat Mar 27 2020
Last revised: April 5, 2020
@author: mouloua ramdane


"""


import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import plkClassifier as plkc
import model as plkm
from sklearn import metrics

import warnings

with warnings.catch_warnings():
	# Uncomment the next lines to auto-reload libraries (this causes some problem with pickles in Python 3)
	import seaborn as sns; sns.set()
	import warnings
	warnings.simplefilter(action='ignore', category=FutureWarning)
	import matplotlib.pyplot as plt
	import pandas as pd
	data_dir = 'public_data'          # The sample_data directory should contain only a very small subset of the data
	data_name = 'plankton'
	import numpy as np
	import preprocessing as prep

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
	from scipy.stats import uniform
	from sklearn.ensemble import StackingClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.base import BaseEstimator
	from sklearn.datasets import load_wine

def testplkClassifier(X, Y, model_name, model_list):
	testAssist = plkc.plkAssitClassifier(model_name, model_list, X, Y)
	best_model_name, best_model_list = testAssist.compareModel()
	model_prefinal = testAssist.find_best_param_MODEL(best_model_name, best_model_list)
	print("model_prefinal ", model_prefinal)

	for i in range(len(model_prefinal)):
		M = plkc.Classifier(X,Y)
		M.process(X,Y, model_process = model_list[i] )
		M.cross_validation_Classifier()
		M.training_score_Classifier()

	model_final = testAssist.stacking(model_prefinal)
	print("DEBUT STACKING ")
	M1 = plkc.Classifier(X,Y)
	M1.process(X,Y, model_process = model_final )
	M1.cross_validation_Classifier()
	M1.training_score_Classifier()



def testplkModel(X, Y):
	scoring_function1 = getattr(metrics, "balanced_accuracy_score")
	print(scoring_function1)
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
	"""
	#D = DataManager(data_name, data_dir) # We reload the data with the AutoML DataManager class because this is more convenient

	X_train = D.data['X_train']
	Y_train = D.data['Y_train'].ravel()

	print(len(X_train[0]))
	print(len(X_train))
	print(len(Y_train))"""

	model_nameS = ["ExtraTreesClassifier", "RandomForestClassifier"]
	model_listS = [ ExtraTreesClassifier() ,RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1)]

	X_Random = np.random.rand(10752,203) #105752 lignes et 203 colonnes pour les images
	Y_Random = np.random.randint(7,size=10752) #105752 lignes et 203 colonnes pour les images
	#Data = load_wine()
	#X_Random = Data.data
	#Y_Random = Data.target


	#testplkClassifier(X_Random, Y_Random, model_nameS, model_listS)
	testplkModel(X_Random, Y_Random)