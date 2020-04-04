import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import plkClassifier as plkc
import modelPLK as plkm

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
	from data_manager import DataManager
	from data_io import write
	from model import model
	from libscores import get_metric
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

if __name__=="__main__":
	import matplotlib
	matplotlib.rcParams['backend'] = 'Qt5Agg'
	matplotlib.get_backend()
	D = DataManager(data_name, data_dir) # We reload the data with the AutoML DataManager class because this is more convenient

	model_name = ["Nearest Neighbors", "Random Forest"]
	model_list = [KNeighborsClassifier(1),  RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=19, min_samples_leaf= 7)]


	Prepro = prep.Preprocessor() # we use pre-processing

	X_train = D.data['X_train']
	Y_train = D.data['Y_train'].ravel() 

	A = plkm.plkClassifier()
	#A.fit(X_train, Y_train)

	"""	M = plkc.Classifier(A.xPLK,A.yPLK)
	M.testModel( model_process = A )
	scores = M.cross_validation_Classifier()"""

	metric_name1, scoring_function1 = get_metric()
	res = cross_val_score(A, X_train, Y_train, cv=5 , scoring = make_scorer(scoring_function1))
	print("cross_validation_Classifier:  ", res)