"""
Created on Sat Mar 27 2020
Last revised: Mai 7, 2020
@author: MUSSARD Romain
This program test plkPreprocessing.py

"""



'''A modifier pour réaliser ses propres tests'''
    
model_dir = 'sample_code_submission/'
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
#data_dir = 'public_data'
data_dir = 'public_data_raw_gaiasavers'   #POUR TRAVAILLER SUR RAW DATA
data_name = 'plankton'

'''Fin de modification'''

from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir); 
from sklearn.preprocessing import MinMaxScaler
from libscores import get_metric
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from data_manager import DataManager
import plkPreprocessing as plkp
from plkPreprocessing import Preprocessor


Prepro = Preprocessor()

D = DataManager(data_name, data_dir) # Load data
print("*** Original data ***")
print(D)


D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
D.data['X_test'] = Prepro.transform(D.data['X_test'])
print("*** Transformed data ***")
print(D)


D = DataManager(data_name, data_dir) # Load data
print("***Start Search Best Parameter***")
print(D)
X = D.data['X_train']
Y = D.data['Y_train'].ravel()

print("X shape : ", X.shape)

X = plkp.createNewFeatures(X)
scaler = MinMaxScaler()

print("new X shape : ", X.shape)

print("data : \n", X)


res = plkp.findBestKneighbors(X,Y)
print("best nb features for otuliersDeletion  = ", res)


prep = Preprocessor()
X,Y = Preprocessor.outliersDeletion(X,Y, nbNeighbors=res)
print("X shape after outliers deletion: ", X.shape)


res1 = plkp.findBestSkb(X, Y)
print("best nb features for skb  = ", res1)


res2 = plkp.findBestPca(X, Y, nb_feat=res1)
print("best nb features for pca  = ", res2)

print("best nb features for otuliersDeletion  = ", res)
print("best nb features for skb  = ", res1)
print("best nb features for pca  = ", res2)


D = DataManager(data_name, data_dir) # Load data
print("***Outliers Deletion***")
print(D)
X_test = D.data['X_train']
Y_test = D.data['Y_train'].ravel()


'''On calcul le score après preprocessing'''
X_test, Y_test = Preprocessor.construct_features(X_test, Y_test)

clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
prepro = Preprocessor()
pipe = Pipeline([('prepro', prepro), ('clf', clf)])
metric_name1, scoring_function1 = get_metric()
res = cross_val_score(pipe, X_test, Y_test, cv=5, scoring = make_scorer(scoring_function1))
print(res, " moyenne = ", res.mean())