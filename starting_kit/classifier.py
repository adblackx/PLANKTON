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

class Classifier:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.ytP = None
		self.yvP = None
		self.M= None

	def process(self, x, y, model_process):
		self.M = model(classifier=model_process)
		trained_model_name = model_dir + data_name
		if not(self.M.is_trained):
			self.M.fit(self.x,self.y)

		self.M.save(trained_model_name)                 
		result_name = result_dir + data_name
		self.ytP = self.M.predict(D.data['X_train'])
		self.yvP = self.M.predict(D.data['X_valid'])


	def scoreClassifier(self):
		metric_name1, scoring_function1 = get_metric()
		return cross_val_score(self.M, self.x,self.y, cv=5 ,scoring = make_scorer(scoring_function1))
	def scoringClassifier(self):
		return scoring_function(self.y,self.ytP)



def finBest(model_name, model_list):
    
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

    c1=[]
    c2=[]
    c3=[]
    
    
    for i in np.arange(len(model_list)) :

        M = Classifier(X_train,Y_train)
        M.process(x=X_train,y=Y_train, model_process = model_list[i] )
        scores = M.scoreClassifier()
        c1.append(model_name[i])
        c2.append(scores.mean())
        c3.append(M.scoringClassifier())


    return c1,c2,c3



if __name__=="__main__":
	import matplotlib
	matplotlib.rcParams['backend'] = 'Qt5Agg'
	matplotlib.get_backend()
	D = DataManager(data_name, data_dir, replace_missing=True) # We reload the data with the AutoML DataManager class because this is more convenient
	"""model_name = ["Nearest Neighbors",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "ExtraTreesClassifier"]
	model_list = [
	    KNeighborsClassifier(1),
	    DecisionTreeClassifier(max_depth=10),
	    #RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
	    RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1),
	    MLPClassifier(alpha=1, max_iter=1000),
	    AdaBoostClassifier(),
	    GaussianNB(),
	    QuadraticDiscriminantAnalysis(),
	    ExtraTreesClassifier()
	]"""
	model_name = ["Nearest Neighbors"]
	model_list = [KNeighborsClassifier(1)]

	X_train = D.data['X_train']
	Y_train = D.data['Y_train'].ravel()
	metric_name, scoring_function = get_metric()


	res1, res2, res3 = finBest(model_name, model_list)
	frame = pd.DataFrame(
    	{
                    "Model " : res1,
                    "Cross-Validation ": res2,
                    "train ": res3,
    	}
	)

	print(frame)

	frame[['Cross-Validation ', 'train ']].plot.bar()
	#plt.ylim(0.5, 1)
	plt.ylabel(sklearn_metric.__name__)

	type(plt)
	plt.show()