"""
Created on Sat Mar 27 2020
Last revised: April 5, 2020
@author: mouloua ramdane


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
	from sklearn.linear_model import LogisticRegression
	from sklearn import metrics



class Classifier:
	"""
	Class used to run repetitive models and to score and to cross validate faster
	is it to have shorter codee and less code
	"""
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.ytP = None
		self.yvP = None
		self.M= None

	def process(self, x, y, model_process):
		print("process")
		self.M = model_process
		self.M.fit(self.x,self.y)

		self.ytP = self.M.predict(x)
		self.yvP = self.M.predict(x)

	def testModel(self, model_process):
		self.M = model_process


	def cross_validation_Classifier(self):
		scoring_function1 = getattr(metrics, "balanced_accuracy_score")
		res = cross_val_score(self.M, self.x, self.y, cv=5 , scoring = make_scorer(scoring_function1))
		print("cross_validation_Classifier:  ", res)
		return res

	def training_score_Classifier(self):
		scoring_function1 = getattr(metrics, "balanced_accuracy_score")
		res = scoring_function1(self.y,self.ytP)
		print("training_score_Classifier:  " , res)
		return res

class plkAssitClassifier:
	"""
		Runs different models with random parameters.
		This function is supposed to find best models and their best parameters, and to return if we want a stacked model
		it is supposed to be used for plkClassifier
	    Parameters
	    ----------
	    model_namePLK: model's liste
	    model_listPLK: list of random values for models
	    x : data
	    y : labels
	"""

	def __init__(self, model_namePLK, model_listPLK, x, y):
		self.model_namePLK = model_namePLK
		self.model_listPLK = model_listPLK
		self.x = x
		self.y = y
		self.best_model_namePLK = []
		self.best_model_listPLK = []
		self.model_final = None
		self.seuil = 0.1

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
	    	print("finBest: models runned: " , model_list[i])
	    	M = Classifier(self.x,self.y)
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
	    model_listF: model's parametersd
	  

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
		    if model_nameF[i] == "ExtraTreesClassifier" or model_nameF[i] == "RandomForestClassifier" :
		        self.best_model_namePLK.append(model_nameF[i])
		        self.best_model_listPLK.append(model_listF[i])

		print("compareModel: best models returned ", self.best_model_namePLK )

		frame[['Cross-Validation ', 'train ']].plot.bar()
		#plt.ylim(0.5, 1)
		"""plt.ylabel(sklearn_metric.__name__)

		type(plt)
		plt.show()"""
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

	    clf = RandomizedSearchCV(logistic, distributions, random_state=0, scoring=make_scorer(scoring_function) )
	    search = clf.fit(self.x, self.y)
	    print(search.best_params_)
	    return search


	def doBestModel(self):

		"""
	    This function runs the best models and print the results ( test function )
	    
	    Parameters
	    ----------
	    Model_final: best Model's class
	  

		"""

		Model_final=self.model_final

		best_M = Classifier(self.x, self.y) 
		best_M.process(x=self.x, y=self.y, model_process = Model_final)

		#print("Cross-Validation ", best_M.cross_validation_Classifier().mean())
		#print("Training score for the balanced_accuracy_score metric: ", best_M.training_score_Classifier())
		score = best_M.cross_validation_Classifier()
		print('doBestModel:  \nCV score (95 perc. CI): %0.2f (+/- %0.2f)' % (score.mean(), score.std() * 2))
		print('doBestModel:  Training train score for the', metric_name, 'metric = %5.4f' % best_M.training_score_Classifier())
		print('doBestModel:  Ideal score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_train, Y_train))


	def find_best_param_MODEL(self, model_name, model_list):
			"""
		    This function runs the best models and print the results ( test function )
		    
		    Parameters
		    ----------
		    model: best Model's class

			"""


			#logistic = RandomForestClassifier()
			res = []
			for i in range(len(model_list)):

				print("find_best_param_MODEL: runs methode", model_list[i])
				logistic = model_list[i]


				# first tests
				#distributions = dict( n_estimators=np.arange(0,200) , min_samples_split=[0,1,2],random_state=np.arange(0,20,1), min_samples_leaf=np.arange(0,20,1) )
				#{'random_state': 19, 'n_estimators': 196, 'min_samples_split': 2, 'min_samples_leaf': 7} 
				# bad results

				#distributions = dict( n_estimators=np.arange(10,200) , min_samples_split=[2],random_state=[0,1,2] )
				#distributions = dict( n_estimators=np.arange(150,200) , min_samples_split=[2],random_state=[0,1,2] )
				if model_name[i] == "ExtraTreesClassifier" or model_name[i] == "RandomForestClassifier" :
					print(model_name[i])
					distributions = dict( n_estimators=np.arange(1,2) , min_samples_split=[2],random_state=[2] )
				else :
					print("pas encore pris en compte.... il n'y a que deux modeles interessant pour le moment")
				search = self.best_param_MODEL(logistic, distributions)
				#print(search.best_params_)
				m = model_list[i]
				for v in search.best_params_:
					m.v = search.best_params_[v]
					print(search.best_params_[v])
				res.append(m)
				#print(search.cv_results_)
			return res

	def voting(self, model_list):
		"""
	    This function runs the best models and print the results ( test function )
	    
	    Parameters
	    ----------
	    Model_final: best Model's class

		"""
		model_listS= []
		for i in np.arange(len(model_list)):
			st = 'rf' + str(i)
			model_listS.append( (st, model_list[i] ))



		print("voting: runs methode")

		#clf = StackingClassifier(estimators=model_listS, final_estimator=LogisticRegression())
		clf = VotingClassifier(estimators=model_listS, voting='soft')
		self.model_final = clf

		return clf
		#self.doBestModel() # to see the results

class findModel:

	def __init__(self):
		self.model_name = ["ExtraTreesClassifier", "RandomForestClassifier"]
		self.model_list = [ ExtraTreesClassifier() ,RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1)]

	def getModel(self,X,Y):
		testAssist = plkAssitClassifier(self.model_name, self.model_list, X, Y)
		best_model_name, best_model_list = testAssist.compareModel()
		model_prefinal = testAssist.find_best_param_MODEL(best_model_name, best_model_list)

		"""for i in range(len(model_prefinal)):
			M = plkc.Classifier(X,Y)
			M.process(X,Y, model_process = model_list[i] )
			M.cross_validation_Classifier()
			M.training_score_Classifier()"""

		model_final = testAssist.voting(model_prefinal)
		"""print("DEBUT voting ")
		M1 = Classifier(X,Y)
		M1.process(X,Y, model_process = model_final )
		M1.cross_validation_Classifier()
		M1.training_score_Classifier()"""

		return model_final





if __name__=="__main__":
	import matplotlib
	matplotlib.rcParams['backend'] = 'Qt5Agg'
	matplotlib.get_backend()
	# = DataManager(data_name, data_dir) # We reload the data with the AutoML DataManager class because this is more convenient

	model_name = ["Nearest Neighbors", "Random Forest"]
	model_list = [KNeighborsClassifier(1),  
				RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=19, min_samples_leaf= 7)]


	"""model_name = ["Nearest Neighbors",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "ExtraTreesClassifier"]
	model_list = [
	    KNeighborsClassifier(1),
	    DecisionTreeClassifier(max_depth=10),
	    #RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
	    RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=19, min_samples_leaf=7),
	    MLPClassifier(alpha=1, max_iter=1000),
	    AdaBoostClassifier(),
	    GaussianNB(),
	    QuadraticDiscriminantAnalysis(),
	    ExtraTreesClassifier()
	]"""

	"""

	X_train = D.data['X_train']
	Y_train = D.data['Y_train'].ravel()
	scoring_function = getattr(metrics, "balanced_accuracy_score")"""


	#X_train = D.data['X_train']
	#Y_train = D.data['Y_train'].ravel() 

	X_train = np.random.rand(10752,203) #105752 lignes et 203 colonnes pour les images
	Y_train = np.random.randint(7,size=10752) #105752 lignes et 203 colonnes pour les images


	scoring_function = getattr(metrics, "balanced_accuracy_score")

	#compareModel(model_name, model_list)




	#M_Model = model(RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0))
	#doBestModel(M_Model)

	model_nameS = ["ExtraTreesClassifier", "RandomForestClassifier"]
	model_listS = [
    ('rf', ExtraTreesClassifier()),
    #('knb',     KNeighborsClassifier(1),
    ('rfc', RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1)),
    #('rfc1',      MLPClassifier(alpha=1, max_iter=1000)),
    #('rfc2',      GaussianNB()),
    #('rfc3',          QuadraticDiscriminantAnalysis()),
	]



	#testAssist= plkAssitClassifier(model_name, model_list , X_train, Y_train)
	
	#best_model_name, best_model_list = testAssist.compareModel()


	#testAssist= plkAssitClassifier(["Random Forest"], [RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=19, min_samples_leaf= 7)] , X_train, Y_train)
	#model_bestParam = testAssist.find_best_param_MODEL(RandomForestClassifier())

	model_nameS1 = ["ExtraTreesClassifier", "RandomForestClassifier"]
	model_listS1 = [ ExtraTreesClassifier() ,RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1)]
	print("model_listS1 ",model_listS1)
	testAssist= plkAssitClassifier(model_nameS1, model_listS1, X_train, Y_train)

	model_final = testAssist.voting(model_listS1)
	print("DEBUT voting ")
	M1 = Classifier(X_train,Y_train)
	M1.process(X_train,Y_train, model_process = model_final )
	M1.cross_validation_Classifier()
	M1.training_score_Classifier()

	"""
	print("debut test best param")

	M = Classifier(X_train,Y_train)
	M.process(X_train,Y_train, model_process = model_bestParam )SS
	M.cross_validation_Classifier()
	M.training_score_Classifier()
	"""
	#testAssist.doBestModel()



	#testAssist.voting(model_listS)