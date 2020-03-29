import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import plkClassifier

model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'

from sys import path; 
path.append(model_dir); 
path.append(problem_dir); 
path.append(score_dir); 

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

class plkClassifier(BaseEstimator):

	"""
	We are still working in this class, we are trying to find the best way to use it with plkAssitClassifier....
	"""

	'''plkClassifier: a model that using best model from testClassifier'''
	def __init__(self):
		'''We replace this model by others model, should be used for usging best model parameters'''
		self.clf = StackingClassifier( estimators=[('rf', ExtraTreesClassifier()), ('rfc', RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1))], final_estimator=LogisticRegression() )

	def fit(self, X, y):
		''' This is the training method: parameters are adjusted with training data.'''
		return self.clf.fit(X, y)

	def predict(self, X):
		''' This is called to make predictions on test data. Predicted classes are output.'''
		return self.clf.predict(X)

	def predict_proba(self, X):
		''' Similar to predict, but probabilities of belonging to a class are output.'''
		return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes

	def get_classes(self):
		return self.clf.classes_

	def save(self, path="./"):
		pickle.dump(self, open(path + '_model.pickle', "w"))

	def load(self, path="./"):
		self = pickle.load(open(path + '_model.pickle'))
		return self


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

	X_train, Y_train = Prepro.outliersDeletion(D.data['X_train'],D.data['Y_train'])
	X_train = Prepro.fit_transform(X_train, Y_train)


	metric_name, scoring_function = get_metric()



	model_listS = [
    ('rf', ExtraTreesClassifier()),
    #('knb',     KNeighborsClassifier(1),
    ('rfc', RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1)),
    #('rfc1',      MLPClassifier(alpha=1, max_iter=1000)),
    #('rfc2',      GaussianNB()),
    #('rfc3',          QuadraticDiscriminantAnalysis()),
	]