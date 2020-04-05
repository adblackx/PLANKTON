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
import plkPreprocessing as prep
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

class plkClassifier(BaseEstimator):

	"""
	We are still working in this class, we are trying to find the best way to use it with plkAssitClassifier....
	"""

	'''plkClassifier: a model that using best model from testClassifier'''
	def __init__(self ):
		'''We replace this model by others model, should be used for usging best model parameters'''

		#self.clf = VotingClassifier( estimators=[('rfc', RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1))], final_estimator=LogisticRegression() )
		
		pipe_class = Pipeline([
					('preprocessing', prep.Preprocessor() ),
					('classification', RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1))
					])

		self.clf =  RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1) 
		self.clf = pipe_class

		self.xPLK = None
		self.yPLK = None
		self.num_train_samples = 0
		self.num_feat=1
		self.num_labels=1
		#self.prepo = prep.Preprocessor()



	def fit(self, X, y):
		''' This is the training method: parameters are adjusted with training data.'''
		#dm = plkc.findModel()
		#self.clf = dm.getModel(X,y) # on trouve le meilleur model ici quand on fit sur les donnees

		self.num_train_samples = X.shape[0]
		if X.ndim>1: 
			self.num_feat = X.shape[1]

		num_train_samples = y.shape[0]
		if y.ndim>1: 
			self.num_labels = y.shape[1]
		if (self.num_train_samples != num_train_samples):
			print("fit: THERE A PROBLEM WITH THE DATA")

		#X1, y1 = self.prepo.outliersDeletion(X, y)
		#X1 = self.prepo.fit_transform(X1, y1)
		x1,y1 = prep.Preprocessor.outliersDeletion(X,y)
		self.clf.fit(x1, y1)

	def predict(self, X):
		''' This is called to make predictions on test data. Predicted classes are output.'''

		if X.ndim>1: 
			num_feat = X.shape[1]
		if (self.num_feat != num_feat):
			print("ARRGH: number of features in X does not match training data!")

		#X1 = self.prepo.transform(X)
		return self.clf.predict(X)

	def predict_proba(self, X):
		''' Similar to predict, but probabilities of belonging to a class are output. '''
		#X1 = self.prepo.transform(X)
		return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes

	def get_classes(self):
		return self.clf.classes_

	def save(self, path="./"):
		pickle.dump(self, open(path + '_model.pickle', "w"))

	def load(self, path="./"):
		self = pickle.load(open(path + '_model.pickle'))
		return self
