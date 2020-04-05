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

from sys import path 
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from data_manager import DataManager
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import plkClassifier as plkc
import modelPLK as plkm
from libscores import get_metric
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


class Preprocessor(BaseEstimator):
    '''n_components == nb_feat ==> no pca'''
    def __init__(self):
        n_components = 19
        nb_feat = 197
        self.skb = SelectKBest(chi2, k = nb_feat)
        self.pca = PCA(n_components)

    def fit(self, X, Y):
        self.skb = self.skb.fit(X,Y)
        X_temp = self.skb.transform(X) #car si non pca n'aura pas les bonnes dimensions
        self.pca = self.pca.fit(X_temp)
        return self

    def fit_transform(self, X, Y):
        X_res = self.skb.fit_transform(X,Y)
        X_res = self.pca.fit_transform(X_res)
        return X_res

    def transform(self, X):
        X_res = self.skb.transform(X)
        X_res = self.pca.transform(X_res)
        return X_res
    
    def outliersDeletion(X, Y, nbNeighbors = 3):
        lof = LocalOutlierFactor(n_neighbors=nbNeighbors)
        decision = lof.fit_predict(X)
        return X[(decision==1)],Y[(decision==1)]

def max_indice(x):
        maxIndice = 0
        for i in range(len(x)):
            if x[i]>x[maxIndice]:
                maxIndice = i
        return maxIndice


def findBestSkb(X, Y):
    score = []
    nb_features = []
    for i in range(191,203,2):
        Xsauv = np.copy(X)
        Ysauv = np.copy(Y)
        clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
        skb = SelectKBest(chi2, k = i)
        pipe = Pipeline([('skb', skb), ('clf', clf)])
        metric_name1, scoring_function1 = get_metric()
        res = cross_val_score(pipe, Xsauv, Ysauv, cv=2 , scoring = make_scorer(scoring_function1))
        score.append(np.mean(res))
        nb_features.append(i)
    it = max_indice(score)
    print(score)
    return nb_features[it]  


def findBestPca(X, Y, nb_feat = 197):
    score = []
    nb_features = []
    for i in range(1,20,2):
        Xsauv = np.copy(X)
        Ysauv = np.copy(Y)
        clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
        skb = SelectKBest(chi2, k= nb_feat)
        pipe = Pipeline([('skb', skb), ('pca', PCA(i)), ('clf', clf)])
        metric_name1, scoring_function1 = get_metric()
        res = cross_val_score(pipe, Xsauv, Ysauv, cv=2 , scoring = make_scorer(scoring_function1))
        score.append(np.mean(res))
        nb_features.append(i)
    it = max_indice(score)
    print(score)
    return nb_features[it]


def findBestKneighbors(X, Y):
    score = []
    nb_features = []
    for i in range(1,10,2):
        Xsauv = np.copy(X)
        Ysauv = np.copy(Y)
        Xsauv, Ysauv = Preprocessor.outliersDeletion(Xsauv, Ysauv,nbNeighbors=i)
        clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
        metric_name1, scoring_function1 = get_metric()
        res = cross_val_score(clf, Xsauv, Ysauv, cv=2 , scoring = make_scorer(scoring_function1))
        score.append(np.mean(res))
        nb_features.append(i)
    it = max_indice(score)
    print(score)
    return nb_features[it]
   
if __name__=="__main__":
    data_dir = 'public_data'
    data_name = 'plankton'
    
    Prepro = Preprocessor()
    
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
    
    D = DataManager(data_name, data_dir) # Load data
    D.data['X_train'], D.data['Y_train'] = Preprocessor.outliersDeletion(D.data['X_train'],D.data['Y_train'])
    print("***Outliers Deletion***")
    print(D)
    X = D.data['X_train']
    Y = D.data['Y_train']
    Xsauv = np.copy(X)
    Ysauv = np.copy(Y)
    
    '''
    res = findBestKneighbors(X,Y)
    print("best nb features for otuliersDeletion  = ", res)
    
    prep = Preprocessor()
    X,Y = Preprocessor.outliersDeletion(X,Y, nbNeighbors=res)
    
    res1 = findBestSkb(X, Y)
    print("best nb features for skb  = ", res1)
    
    res2 = findBestPca(X, Y, nb_feat=res1)
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
    
    
    

    
    
