"""
Created on Sat Mar 27 2020
Last revised: April 27, 2020
@author: Mouloua Ramdane

Ajout du model, on utilise la classe findModel dans plkClassifier pour trouver notre modele
on set notre cle a None, on trouve notre model quand on fit par rapport a nos données
RAW DATA

Model:
- the model uses a Pipeline
- the model uses a pre-processing
- the model uses a outliersDeletion
- function fit : - the model fit a first time
				 - the model find best parameters
				 - the model uses GridSearchCV
				 - the model uses cross_val_score
				 - the model uses a metrics : "balanced_accuracy_score" 

- we selected previously three models:
				- the model runs and find best parameters for each model in findMODEL
				- the best parameters are find thanks to GridSearchCV
				- we compare the cross_val_score and the metric
				- the model is supposed to find best hyperparameters and it is supposed to be 
				  runned only one time, and then it returns each model with best parameters 
				  for the data
last updates:
	- nous avons mis des commentaires partout pour eclaircir comment la classe fonctionne
	- ON PEUT A PRESENT CHOISIR	:
		- UN ENSEMBLE DE MODELES
		- UN ENSEMBLE DE PARAMETERES A TESTER POUR CHAQUE MODELE
		- LE TOUT SERA COUPLE EN UN VOTING
	- model_name et model_list SONT UTILISE QUE POUR HARDCODER SI ON VEUT POUR ACCELERER LES TESTS
	MAIS DANS LA PLUPART DU TEMPS ON CHERCHE QUAND MEME LES MEILLEURS PARAMETRES SI isFitted = FALSE
			
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
from sys import path
from sys import argv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class model(BaseEstimator):

	test = None
	'''Classifier: This is a Model using a Pipeline (preprocessing + model)
		As we describe it upper:
		 '''

	'''self.model_name = ["ExtraTreesClassifier", 
							"RandomForestClassifier",
							"DecisionTreeClassifier"]
		self.model_list = [ ExtraTreesClassifier(n_estimators=130) ,
							RandomForestClassifier(n_estimators=98, max_depth=None, min_samples_split=2, random_state=1),
							DecisionTreeClassifier(max_depth=13, max_features = 'sqrt',random_state=42)]
	'''

	def __init__(self, isFitted=False, model_name = ["ExtraTreesClassifier","RandomForestClassifier"],
			model_list = [ ExtraTreesClassifier(n_estimators=250, min_samples_split=2,random_state=2) ,
			RandomForestClassifier(n_estimators=200, min_samples_split=2,random_state=2)], 
			clf=None, dict_list=None):

		"""
		We we call the constructor of this class, if isFitted = False then we generate a model
		if is True, we load a saved model
		IMPORTANT: The model self.clf is  set to None, because we create it when we FIT, not before
		the clf can be generated thanks to plkClassifier 
	
		Parameters
		----------
		isFitted: Boolean that used to see if we generate the model or if we load it 
				in the function fit
		model_name : liste of model's name that will be tested
		model_list : liste of model
		dict_list : contains a list of dict, each dict contains:
				- model's name, 
				-list of parameters names, 
				-and a list of parameter's value
		"""
		
		self.clf = clf
		self.num_train_samples = 0
		self.num_feat=1
		self.num_labels=1
		self.prepo = prep.Preprocessor()
		self.isFitted = isFitted
		self.model_name = model_name
		self.model_list = model_list
		self.dict_list = dict_list


	def fit(self, X, y):
		"""
		This is the training method: parameters are adjusted with training data.
		In fact, we start by checking that the data are the same.
		Then we check if THE HYPERPARAMETERS are already exists (by a precedent call of this function)
		If is is the first time, then we use findModel in plkClassifier
		findModel returns a voting model, it combines three models, with their best hyper parameters
		find thanks to GridSearchCV, cross_val_score, "balanced_accuracy_score" use by plkAssitClassifier

		Parameters
		----------
		X: Data
		y: label (or classe)
			
		This function check if 
		"""

		#checking the data are corrects
		self.num_train_samples = X.shape[0]
		if X.ndim>1: 
			self.num_feat = X.shape[1]

		num_train_samples = y.shape[0]
		if y.ndim>1: 
			self.num_labels = y.shape[1]
		if (self.num_train_samples != num_train_samples):
			print("model.py, fit: THERE IS A PROBLEM WITH THE DATA")

		"""if not self.isFitted :
			#hardcoded version
			# this hardcode work on

			x1,y1 = prep.Preprocessor.construct_features(X,y)
			a = plkc.assistModel(x1,y1,prepo=prep.Preprocessor())

			voting_model = a.voting(self.model_list)
			pipe_class = Pipeline([
						('preprocessing', self.prepo ),
						('voting', voting_model)
						])

			self.clf = pipe_class
			self.clf.fit(x1, y1)	"""

		if not self.isFitted :

			# We use here preprocessing
			x1,y1 = prep.Preprocessor.construct_features(X,y)

			class_to_find_voting_model = plkc.assistModel(x1,y1,prepo=prep.Preprocessor())

			if self.dict_list == None: # if we don't set a dictionary we use this one as default
				d1 = { 
					 "name" :"ExtraTreesClassifier",
					 "param_name" : ["n_estimators", "min_samples_split","random_state"],
					 "param_val" : [[200,250], [2], [2]]
				}

				d2 = { 
					 "name" :"RandomForestClassifier",
					 "param_name" : ["n_estimators", "min_samples_split","random_state"],
					 "param_val" : [[200,250], [2], [2]]
				}
				self.dict_list = [d1, d2]
			
			class_to_find_voting_model.setModelsPrepro(self.model_name, self.model_list, self.dict_list, setBest=True )
			voting_model = class_to_find_voting_model.getModelPrepro()
			pipe_class = Pipeline([
						('preprocessing', self.prepo ),
						('voting', voting_model)
						])

			self.clf = pipe_class
			self.clf.fit(x1, y1)
			self.isFitted = True # so we generate the best model here



		else: 
			# it is the case that the best model is generated, so we load it, no need to fit again 
			# it takes too long time...
			x1,y1 = prep.Preprocessor.construct_features(X,y)

			self.clf.fit(x1, y1)
		print("FIT MON MODEL")


	def predict(self, X):
		"""
		This function is called to make predictions on test data. 

		Parameters
		----------
		X: The data

		Returns
		------
		Predicted class (labels)

		"""
		# we check the data here if they are the same that we have fitted on...

		if X.ndim>1: 
			num_feat = X.shape[1]
		if (self.num_feat != num_feat): 
			print("ARRGH: number of features in X does not match training data!")

		print("PREDICT ")
		return self.clf.predict(X)

	def predict_proba(self, X):
		"""
		This function is called to make predictions on test data. 
		Similar to predict, but probabilities of belonging to a class are output
		The classes are in the order of the labels returned by get_classes

		Parameters
		----------
		X: The data

		Returns
		------
		probabilities by class (labels)

		"""
		print("PREDICT PROBA ")

		return self.clf.predict_proba(X) 

	def get_classes(self):
		"""We just return the classes here"""
		return self.clf.classes_

	def save(self, path="./"):
		"""
		Save a model using pickle in a specified path

		Parameters
		----------
		path: a path where the model will be saved


		"""
		print("save ", path, " ",'_model.pickle' )
		file = open(path + '_model.pickle', "wb")
		pickle.dump(self, file)
		file.close()


	def load(self, path="./"):
		"""
		load a model from a specified path

		Parameters
		----------
		path: a path where the model is

		"""
		modelfile = path + '_model.pickle'
		if isfile(modelfile):
			with open(modelfile, 'rb') as f:
				self = pickle.load(f)
			print("Model reloaded from: " + modelfile)

		return self

def testclassCVM(X, Y, model):

	"""
	This function runs classCVM and print results

	Parameters
	----------
	X: Data
	Y: model's label (classes)
	model: the model
	"""
	print("testclassCVM : BEGIN")
	cvm = plkc.classCVM(X,Y)
	cvm.process(X,Y,model)
	cvm.cross_validation_Classifier()
	cvm.training_score_Classifier()
	print("testclassCVM : END")

def testModel(X, Y):
	"""
	This function runs test the model with default values and print results

	Parameters
	----------
	X: Data
	Y: model's label (classes)
	"""

	print("testModel : BEGIN")
	a = model()
	a.fit(X,Y)
	testclassCVM(X,Y,a)

	#if you want to print
	"""scoring_function1 = getattr(metrics, "balanced_accuracy_score")
	res = cross_val_score(a, X, Y, cv=5 , scoring = make_scorer(scoring_function1))
	print("cross_validation_Classifier:  ", res)
	print("cross_validation_Classifier (moyenne)  ", res.mean())	"""
	print(a)
	print("testModel : END")

def testModel1(X, Y):
	"""
	This function runs test A way to run the model with customised parameters

	Parameters
	----------
	X: Data
	Y: model's label (classes)
	"""
	d1 = { 
					 "name" :"ExtraTreesClassifier",
					 "param_name" : ["n_estimators", "min_samples_split","random_state"],
					 "param_val" : [[200,250], [2], [2]]
				}

	d2 = { 
		 "name" :"RandomForestClassifier",
		 "param_name" : ["n_estimators", "min_samples_split","random_state"],
		 "param_val" : [[200,250], [2], [2]]
	}

	d = [d1,d2]
	model_name = ["ExtraTreesClassifier","RandomForestClassifier"]
	model_list = [ ExtraTreesClassifier(n_estimators=250, min_samples_split=2,random_state=2) ,
			RandomForestClassifier(n_estimators=200, min_samples_split=2,random_state=2)]

	a = model(model_name=model_name,model_list = model_list, dict_list = d, isFitted=False )
	a.fit(X,Y)
	testclassCVM(X,Y,a)

if __name__=="__main__":
	"""data_dir = 'public_data_raw_gaiasavers'          # POUR TRAVAILLER SUR RAW DATA
	data_name = 'plankton'
	D = DataManager(data_name, data_dir) 

	X_train = D.data['X_train']
	Y_train = D.data['Y_train'].ravel()

	print(len(X_train[0]))
	print(len(X_train))
	print(len(Y_train))"""

	#TEST WITH RANDOM DATA
	X_train = np.random.randint(low=0, high=255, size=(203, 10000))
	Y_train = np.random.randint(7,size=203) 
	#Test1
	#testModel(X_train, Y_train)

	#Test2
	testModel1(X_train, Y_train)

