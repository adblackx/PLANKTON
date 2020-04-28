"""
Created on Sat Mar 27 2020
Last revised: April 18, 2020
@author: Mouloua Ramdane

Correctifs de bugs
Ajout de Voting a la place de stacking
Ajout d'un seuil, qui n'est pas utilsé pour le moment
RAW DATA

"""

other_files = 'other_files/'

from sys import path; 
path.append(other_files);


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
	from sklearn.ensemble import VotingClassifier
	#from sklearn.ensemble import StackingClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn import metrics
	from sklearn.model_selection import GridSearchCV
	import plkPreprocessing as prep
	from sklearn.pipeline import Pipeline


class classCVM:
	"""
	ClassCVM :
	- CV -> crossvlidation
	- M -> metric
	Class used :
	- to use a model
	- to score and to cross validate
	
	"""
	def __init__(self, x, y):
		"""
		This function is called set up the class, 
		and to associate the data and the label. 

		Parameters
		----------
		x: The data
		y: The label (classes)

		"""

		self.x = x
		self.y = y
		self.ytP = None
		self.yvP = None
		self.M= None

	def process(self, x, y, model_process):
		"""
		This function is called:
		-to make predictions. 
		-to set the model
		Parameters
		----------
		x: The data
		y: The label (classes)
		model_process: the model

		"""
		print("process")
		self.M = model_process
		self.M.fit(self.x,self.y)

		self.ytP = self.M.predict(x)
		self.yvP = self.M.predict(x)

	def cross_validation_Classifier(self):
		"""
		This function is called:
		-to make cross valdiation
		-to print the cross validation
		-to print the mean


		Returns
		----------
		This function returns the cross validation array

		"""

		scoring_function1 = getattr(metrics, "balanced_accuracy_score")
		res = cross_val_score(self.M, self.x, self.y, cv=5 , scoring = make_scorer(scoring_function1))
		print("cross_validation_Classifier:  ", res)
		print("cross_validation_Classifier (moyenne)  ", res.mean())
		return res

	def training_score_Classifier(self):
		"""
		This function is called:
		-to make metrics
		-to print the metrics

		Returns
		----------
		This function returns the metrics using 
		balanced_accuracy_score

		"""
		scoring_function1 = getattr(metrics, "balanced_accuracy_score")
		res = scoring_function1(self.y,self.ytP)
		print("training_score_Classifier:  " , res)
		return res

class assistModel:
	"""
		Runs different models with different parameters.
		This function is supposed to find best models and their best parameters, 
		and to return if we want a voting model
		it is supposed to be used for 
	    Parameters
	    ----------
	    model_namePLK: model's liste
	    model_listPLK: list of random values for models
	    x : data
	    y : labels
	"""

	def __init__(self, x, y, alpha=0.1, prepo=prep.Preprocessor()):
		self.model_namePLK = None
		self.model_listPLK = None
		self.x = x
		self.y = y
		self.best_model_namePLK = []
		self.best_model_listPLK = []
		self.model_final = None
		self.seuil = 0.1
		self.alpha = alpha
		self.prepo = prepo

	def setModels(self, model_namePLK, model_listPLK):
		self.model_namePLK = model_namePLK
		self.model_listPLK = model_listPLK

	def setModelsPrepro(self, model_namePLK, model_listPLK):
		res = []
		for i in np.arange(len(model_namePLK)):
			res.append(Pipeline([
						('preprocessing', self.prepo ),
						(model_namePLK[i], model_listPLK[i])
						]))

		self.model_namePLK = model_namePLK
		self.model_listPLK = res

	def getModelPrepro(self):

		best_model_name, best_model_list = self.compareModel()
		model_prefinal = self.find_best_param_MODEL_Prepro(best_model_name, best_model_list)

		print("DEBUT TEST POUR CHAQUE MODEL OPTIMISE")
		'''for i in range(len(model_prefinal)):
			print("MODEL NUMERO", i)
			M = classCVM(X,Y)
			M.classCVM(X,Y, classCVM = self.model_list[i] )
			M.cross_validation_Classifier()
			M.training_score_Classifier()'''

		model_final = self.voting(model_prefinal)
		'''print("DEBUT voting ")
		M1 = classCVM(X,Y)
		M1.classCVM(X,Y, classCVM = model_final )
		M1.cross_validation_Classifier()
		M1.training_score_Classifier()
		A = M1.cross_validation_Classifier()
		print("CV VOTING: ", A.mean())
		print("metric VOTING: ", M1.training_score_Classifier() )'''

		return model_final

	def getModel(self):

		best_model_name, best_model_list = self.compareModel()
		model_prefinal = self.find_best_param_MODEL(best_model_name, best_model_list)

		print("DEBUT TEST POUR CHAQUE MODEL OPTIMISE")
		'''for i in range(len(model_prefinal)):
			print("MODEL NUMERO", i)
			M = classCVM(X,Y)
			M.classCVM(X,Y, classCVM = self.model_list[i] )
			M.cross_validation_Classifier()
			M.training_score_Classifier()'''

		model_final = self.voting(model_prefinal)
		'''print("DEBUT voting ")
		M1 = classCVM(X,Y)
		M1.classCVM(X,Y, classCVM = model_final )
		M1.cross_validation_Classifier()
		M1.training_score_Classifier()
		A = M1.cross_validation_Classifier()
		print("CV VOTING: ", A.mean())
		print("metric VOTING: ", M1.training_score_Classifier() )'''

		return model_final

	def finBest(self):
	    
	    """
	    Runs different models with random parameters.
	    For each model we calculate the cross-valdiation and training performance
	    
	    Parameters
	    ----------
	    model_name: model's liste
	    model_list: list of random values for models
	    
	    
	    Returns
	    ------
	    C1: model name
	    C2: cross-validation score
	    C3: training performance
	    
	    """

	    model_name = self.model_namePLK 
	    model_list = self.model_listPLK 


	    c1=[]
	    c2=[]
	    c3=[]
	    
	    for i in np.arange(len(model_list)) :
	    	print("finBest: models runned: ")
	    	M = classCVM(self.x,self.y)
	    	M.process(x=self.x,y=self.y, model_process = model_list[i] )
	    	scores = M.cross_validation_Classifier()
	    	c1.append(model_name[i])
	    	c2.append(scores.mean())
	    	c3.append(M.training_score_Classifier())

	    return c1,c2,c3

	def compareModel(self):

		"""
	    This function runs models with default paremeters to see the perforemances on the data
	    
	    Parameters
	    ----------
	    model_nameF: model's name
	    model_listF: model's parameters
	  

	    Returns
	    ------
	    search: the best best models and the the parameters

	    """
		model_nameF = self.model_namePLK
		model_listF = self.model_listPLK
		res1, res2, res3 = self.finBest()

		frame = pd.DataFrame(
	    	{
	                    "Model " : res1,
	                    "Cross-Validation ": res2,
	                    "train ": res3,
	    	}
		)

		print(frame)


		for i in range(len(res1)):
		    #if res2[i] > seuil and res3[i]> seuil:
		    if model_nameF[i] == "ExtraTreesClassifier" or model_nameF[i] == "RandomForestClassifier" or model_nameF[i] == "DecisionTreeClassifier" :
		        self.best_model_namePLK.append(model_nameF[i])
		        self.best_model_listPLK.append(model_listF[i])

		print("compareModel: best models returned ", self.best_model_namePLK )

	
		"""
		#UNCOMMENT TO PLOT BAR !!!!!
		frame[['Cross-Validation ', 'train ']].plot.bar()
		
		plt.ylim(0.5, 1)
		plt.ylabel(sklearn_metric.__name__)

		type(plt)
		plt.show()
		"""

		return self.best_model_namePLK, self.best_model_listPLK

	def best_param_MODEL(self, logistic, distributions): 
	    """
	    This function finds the best parameters for the RandomizedSearchCV model and returns the best parameters
	    it uses RandomizedSearchCV ( we can also use gridSearch)
	    
	    Parameters
	    ----------
	    logistic: model's name
	    distributions: dictionary of the different parameters of the model that will be tested
	  

	    Returns
	    ------
	    search: the best parameters

	    """
	    print("best_param_MODEL: ")
	    scoring_function = getattr(metrics, "balanced_accuracy_score")

	    # uncomment to use RandomizedSearchCV 
	    #clf = RandomizedSearchCV(logistic, distributions, random_state=0, scoring=make_scorer(scoring_function) ) 
	    clf = GridSearchCV(logistic, distributions, scoring=make_scorer(scoring_function) )
	    #print(clf)
	    search = clf.fit(self.x, self.y)
	    print(search.best_params_)
	    return search

	def doBestModel(self):

		"""
	    This function runs the best models and print the results ( test function )
	    
		"""

		Model_final=self.model_final
		best_M = classCVM(self.x, self.y) 
		best_M.process(x=self.x, y=self.y, model_process = Model_final)
		#print("Cross-Validation ", best_M.cross_validation_Classifier().mean()) #uncomment to print
		#print("Training score for the balanced_accuracy_score metric: ", best_M.training_score_Classifier()) # uncommenter to print
		score = best_M.cross_validation_Classifier()
		print('doBestModel:  \nCV score (95 perc. CI): %0.2f (+/- %0.2f)' % (score.mean(), score.std() * 2))
		print('doBestModel:  Training train score for the', metric_name, 'metric = %5.4f' % best_M.training_score_Classifier())
		print('doBestModel:  Ideal score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_train, Y_train))

	def find_best_param_MODEL(self, model_name, model_list):
			"""
			This function runs the best models and print the results ( test function )
			For each model we calculate the cross-valdiation and training performance

			Parameters
			----------
			model_name: model's liste
			model_list: list of random values for models

			Returns
			------
			res: best parameters for each models
			"""
			res = []
			for i in range(len(model_list)):
				#print("find_best_param_MODEL: runs methode", model_list[i])
				logistic = model_list[i]
				if model_name[i] == "ExtraTreesClassifier" or model_name[i] == "RandomForestClassifier" :
					distributions = dict( n_estimators=[98,125] , min_samples_split=[2], random_state=[2] )
					# a way to use alpha
					#distributions = dict( n_estimators=np.arange(98-int(self.alpha*20), 125 + int(self.alpha*20)) , min_samples_split=[2], random_state=[2] )
				elif model_name[i] == "DecisionTreeClassifier" :
					distributions = dict( max_depth=[13] , max_features=['sqrt'], random_state=[42] )
				else:
					print("pas encore pris en compte.... il n'y a que deux modeles interessant pour le moment")
				search = self.best_param_MODEL(logistic, distributions)
				m = model_list[i]
				for v in search.best_params_:
					t = m.__dict__[v]
					m.__dict__[v] = search.best_params_[v]
					t = m.__dict__[v]
				res.append(m)
			return res

	def paramtoDict(self, param,value, name):
		a ="dict("
		for i in range(len(param)):
			a+= (name+"__"+str(param[i])+"="+str(value[i])+",")
		a+=")"

		return a

	def find_best_param_MODEL_Prepro(self, model_name, model_list):
			"""
			This function runs the best models and print the results ( test function )
			For each model we calculate the cross-valdiation and training performance

			Parameters
			----------
			model_name: model's liste
			model_list: list of random values for models

			Returns
			------
			res: best parameters for each models
			"""
			res = []
			for i in range(len(model_list)):
				#print("find_best_param_MODEL: runs methode", model_list[i])
				logistic = model_list[i]
				name = model_list[i].steps[1][0]
				if model_name[i] == "ExtraTreesClassifier" or model_name[i] == "RandomForestClassifier" :
					param = ["n_estimators", "min_samples_split","random_state"]
					p1 = [100]
					value = [p1, [2], [2]]
					distributions = eval(self.paramtoDict(param,value,name))

					'''distributions = dict( model__n_estimators=[98,125] , 
										model__min_samples_split=[2], 
										model__random_state=[2] )'''
					# a way to use alpha
					#distributions = dict( n_estimators=np.arange(98-int(self.alpha*20), 125 + int(self.alpha*20)) , min_samples_split=[2], random_state=[2] )
				elif model_name[i] == "DecisionTreeClassifier" :
					name = model_list[i].steps[1][0]
					param = ["max_depth", "max_features","random_state"]
					value = [[13], ['sqrt'], [42]]
					distributions = eval(self.paramtoDict(param,value,name))
				else:
					print("pas encore pris en compte.... il n'y a que deux modeles interessant pour le moment")
				search = self.best_param_MODEL(logistic, distributions)
				m = model_list[i][name]	
				toSplit= name+'__'		
				for v in search.best_params_:
					deb,fin = v.split(toSplit, 1) # conversion to get the name
					t = m.__dict__[fin]
					m.__dict__[fin] = search.best_params_[v]
					t = m.__dict__[fin]
				res.append(m)
			return res

	def voting(self, model_list):

		"""
	    This function runs the best models and print the results 

		Parameters
		----------
		model_list: list of best hyperparaetmers values for models

		Returns
		------
		clf: returns voting models with best parameters for each model
		"""
		model_listS= []
		for i in np.arange(len(model_list)):
			st = 'rf' + str(i)
			model_listS.append( (st, model_list[i] ))


		print("voting: Runs methode")

		clf = VotingClassifier(estimators=model_listS, voting='soft')
		self.model_final = clf

		return clf

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
	cvm = classCVM(X,Y)
	cvm.process(X,Y,model)
	cvm.cross_validation_Classifier()
	cvm.training_score_Classifier()
	print("testclassCVM : END")

def testAssistModel(X, Y, model_name, model_list):

	"""
	This function runs ssistModel and print results

	Parameters
	----------
	X: Data
	Y: model's label (classes)
	model_name: a string list of model
	model_list: a list of model and their parameters
	"""
	print("testAssistModel : BEGIN")
	a = assistModel(X,Y)
	a.setModels(model_name, model_list)
	model = a.getModel()
	model.fit(X,Y)

	print("testAssistModel : END")


if __name__=="__main__":
	#UNCOMMENT AND USE IT TO WORK FOR REAL DATA 

	"""data_dir = 'public_data_raw_gaiasavers'          # POUR TRAVAILLER SUR RAW DATA
	data_name = 'plankton'
	D = DataManager(data_name, data_dir) 

	X_train = D.data['X_train']
	Y_train = D.data['Y_train'].ravel()

	print(len(X_train[0]))
	print(len(X_train))
	print(len(Y_train))"""


	X_train = np.random.rand(203,10000) #105752 lignes et 203 colonnes pour les images
	Y_train = np.random.randint(7,size=203) #105752 lignes et 203 colonnes pour les images
	"""testclassCVM(X_train, Y_train,ExtraTreesClassifier() )

	model_nameS = ["ExtraTreesClassifier", "RandomForestClassifier"]
	model_listS = [ ExtraTreesClassifier() ,RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1)]
	testAssistModel(X_train, Y_train,model_nameS, model_listS)"""
	a = assistModel(X_train, Y_train)
	a.setModels(["ExtraTreesClassifier"], [ExtraTreesClassifier()])

	pipe_class = [Pipeline([
						('preprocessing', prep.Preprocessor() ),
						('model', ExtraTreesClassifier())
						])]

	#model_prefinal = a.find_best_param_MODEL_Prepro(["ExtraTreesClassifier"], pipe_class)

	a.setModelsPrepro(["ExtraTreesClassifier","RandomForestClassifier"], [ ExtraTreesClassifier() ,RandomForestClassifier()])

	voting_model = a.getModelPrepro()

	#print(Pipeline)