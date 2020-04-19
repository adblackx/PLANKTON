"""
Created on Sat Mar 27 2020
Last revised: April 4, 2020
@author: Team Plankton
This program preprocessed data for classification.
Actually this program handle pca, features selection and outliers deletion

Last update (April 4):
    - Creation of findBestSkb
    - Creation of findBestPca
    - Creation of findBestkneighbors
"""
model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir); 

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import plkClassifier as plkc
import model as plkm
from libscores import get_metric
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from data_manager import DataManager
from sklearn.preprocessing import StandardScaler
import plkPreprocessing as plkp
from plkPreprocessing import Preprocessor
from sample_code_submission.plkPreprocessing import binariseImage
#data_dir = 'public_data'
data_dir = 'public_data_raw_gaiasavers'          # POUR TRAVAILLER SUR RAW DATA
data_name = 'plankton'

#Prepro = Preprocessor()

'''
D = DataManager(data_name, data_dir) # Load data
print("*** Original data ***")
print(D)

#Prepro.fit(D.data['X_train'], D.data['Y_train'])
#D.data['X_train'] = Prepro.transform(D.data['X_train'])
D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
D.data['X_test'] = Prepro.transform(D.data['X_test'])
D.feat_name = np.array(['PC1', 'PC2'])
D.feat_type = np.array(['Numeric', 'Numeric'])
print("*** Transformed data ***")
print(D)

'''

D = DataManager(data_name, data_dir) # Load data
D.data['X_train'], D.data['Y_train'] = Preprocessor.outliersDeletion(D.data['X_train'],D.data['Y_train'])
print("***Outliers Deletion***")
print(D)
X = D.data['X_train']
Y = D.data['Y_train']
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#X = binariseImage(X)
print("type de X : ", type(X))

print("X shape : ", X.shape)
#X = X/255
#X = plkp.createNewFeatures(X)
print(X)
print(X.shape)
Xsauv = np.copy(X)
Ysauv = np.copy(Y)
'''
res = plkp.findBestKneighbors(X,Y)
print("best nb features for otuliersDeletion  = ", res)
res = 5
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
'''

clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
prepro = Preprocessor()
pipe = Pipeline([('prepro', prepro), ('clf', clf)])
metric_name1, scoring_function1 = get_metric()
res = cross_val_score(pipe, Xsauv, Ysauv, cv=2 , scoring = make_scorer(scoring_function1))
print(res)

temp = prepro.fit_transform(X,Y)
print("temp shape :", temp.shape)
print(temp)