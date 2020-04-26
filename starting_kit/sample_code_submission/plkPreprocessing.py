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

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA #better than pca
from sklearn.decomposition import KernelPCA #better than the two above
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

class Preprocessor(BaseEstimator):
    '''n_components == nb_feat ==> no pca'''
    def __init__(self):
        n_components = 19 #22 for raw 
        nb_feat = 197 # 3970 for raw
        self.skb = SelectKBest(chi2, k = nb_feat)
        self.pca = KernelPCA(n_components)
        self.scaler = StandardScaler()

    def fit(self, X, Y):
        #X_temp = createNewFeatures(X)
        X_temp = X
        self.skb = self.skb.fit(X_temp,Y)
        X_temp = self.skb.transform(X) #car si non pca n'aura pas les bonnes dimensions
        self.scaler = self.scaler.fit(X_temp)
        X_temp = self.scaler.transform(X_temp)
        self.pca = self.pca.fit(X_temp)
        return self

    def fit_transform(self, X, Y):
        #X_res = createNewFeatures(X)
        X_res = X
        X_res = self.skb.fit_transform(X_res,Y)
        X_res = self.scaler.fit_transform(X_res)
        X_res = self.pca.fit_transform(X_res)
        return X_res

    def transform(self, X):
        #X_res = createNewFeatures(X)
        X_res = X
        X_res = self.skb.transform(X_res)
        X_res = self.scaler.transform(X_res)
        X_res = self.pca.transform(X_res)
        return X_res
    
    def outliersDeletion(X, Y, nbNeighbors = 3): #146 for raw
        sizeb = X.shape[0]
        lof = LocalOutlierFactor(n_neighbors=nbNeighbors, metric = 'correlation')
        decision = lof.fit_predict(X)
        Xres, Yres = X[(decision==1)],Y[(decision==1)]
        print("nb deletion : ", sizeb - Xres.shape[0])
        return Xres, Yres

def max_indice(x):
        maxIndice = 0
        for i in range(len(x)):
            if x[i]>x[maxIndice]:
                maxIndice = i
        return maxIndice


def findBestSkb(X, Y):
    score = []
    nb_features = []
    for i in range(3969,3971,1): #5998,6002,1
        Xsauv = np.copy(X)
        Ysauv = np.copy(Y)
        clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
        skb = SelectKBest(chi2, k = i)
        pipe = Pipeline([('skb', skb), ('clf', clf)])
        scoring_function1 = getattr(metrics, "balanced_accuracy_score")
        res = cross_val_score(pipe, Xsauv, Ysauv, cv=2 , scoring = make_scorer(scoring_function1))
        score.append(np.mean(res))
        nb_features.append(i)
    it = max_indice(score)
    print(score)
    return nb_features[it]  


def findBestPca(X, Y, nb_feat = 3970):
    score = []
    nb_features = []
    #for i in range(21,24,1):
    for i in range(20,30,1):
        Xsauv = np.copy(X)
        Ysauv = np.copy(Y)
        clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
        skb = SelectKBest(chi2, k= nb_feat)
        pipe = Pipeline([('skb', skb), ('std', StandardScaler()), ('pca', KernelPCA(i)), ('clf', clf)])
        scoring_function1 = getattr(metrics, "balanced_accuracy_score")
        res = cross_val_score(pipe, Xsauv, Ysauv, cv=2 , scoring = make_scorer(scoring_function1))
        score.append(np.mean(res))
        nb_features.append(i)
    it = max_indice(score)
    print(score)
    return nb_features[it]


def findBestKneighbors(X, Y):
    score = []
    nb_features = []
    #for i in range(4,7,1):
    for i in range(141,156,1):
        Xsauv = np.copy(X)
        Ysauv = np.copy(Y)
        Xsauv, Ysauv = Preprocessor.outliersDeletion(Xsauv, Ysauv,nbNeighbors=i)
        clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
        scoring_function1 = getattr(metrics, "balanced_accuracy_score")
        res = cross_val_score(clf, Xsauv, Ysauv, cv=2 , scoring = make_scorer(scoring_function1))
        score.append(np.mean(res))
        nb_features.append(i)
    it = max_indice(score)
    print(score)
    return nb_features[it]
 

def binariseImage(X):
    X = X/128
    return X.astype(int)

def createNewFeatures(X):
    X = np.c_[X,X.mean(axis=1)]
    X = np.c_[X,X.std(axis=1)]
    X = np.c_[X,X.var(axis=1)]
    return X
    
    

    
    
