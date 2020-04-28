"""
Created on Sat Mar 27 2020
Last revised: April 4, 2020
@author: MUSSARD Romain
This program preprocessed data for classification.
Actually this program handle pca, features selection and outliers deletion

Update of April 25 :
    - Try other PCA method like IncrementalPca and KernelPca
       chose to keep KernelPca for now

Update of April 27 :
    - Implement CreateNewFeatures : Do the same preprocessing 
      than the preprocessed competition
    
Last update (April 28):
    - Modif of findBestSkb, findBestPca and findBestkneighbors
      Make them search param by dichotomy
"""

import numpy as np
from scipy import ndimage
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
    def __init__(self):
        n_components = 23
        nb_feat = 198 #188
        self.skb = SelectKBest(chi2, k = nb_feat)
        self.pca = KernelPCA(n_components)
        self.scaler = StandardScaler()
        self.feat_size = 204

    def fit(self, X, Y):
        #X_temp = createNewFeatures(np.copy(X))
        X_temp = X
        self.skb = self.skb.fit(X_temp,Y)
        X_temp = self.skb.transform(X) #car si non pca n'aura pas les bonnes dimensions
        self.scaler = self.scaler.fit(X_temp)
        X_temp = self.scaler.transform(X_temp)
        self.pca = self.pca.fit(X_temp)
        self.feat_size = X.shape[1]
        return self

    def fit_transform(self, X, Y):
        #X_res = createNewFeatures(X)
        X_res = X
        if(X.shape[1] != self.feat_size):
            X_res = createNewFeatures(np.copy(X))
        X_res = self.skb.fit_transform(X_res,Y)
        X_res = self.scaler.fit_transform(X_res)
        X_res = self.pca.fit_transform(X_res)
        return X_res

    def transform(self, X):
        #X_res = createNewFeatures(X)    
        X_res = X
        if(X.shape[1] != self.feat_size):
            X_res = createNewFeatures(np.copy(X))
        X_res = self.skb.transform(X_res)
        X_res = self.scaler.transform(X_res)
        X_res = self.pca.transform(X_res)
        return X_res
    
    def outliersDeletion(X, Y, nbNeighbors = 141): #146 for raw
        sizeb = X.shape[0]
        lof = LocalOutlierFactor(n_neighbors=nbNeighbors, metric = 'cityblock')
        decision = lof.fit_predict(X)
        Xres, Yres = X[(decision==1)],Y[(decision==1)]
        print("nb deletion : ", sizeb - Xres.shape[0])
        return Xres, Yres

    def construct_features(X,Y, outliers_deletion = True):
        X = createNewFeatures(X)
        if outliers_deletion :
            X,Y = Preprocessor.outliersDeletion(X,Y)
        return X,Y

def binariseImage(X):
    X = np.where(X>127.5, 1, 0)
    return X

def calcPerimeter(X):
    res = []
    for img in range(len(X)):
        img_2d = np.reshape(X[img], (100, 100))
        img_2d = binariseImage(img_2d)
        sx = ndimage.sobel(img_2d, axis=0, mode='constant')
        sy = ndimage.sobel(img_2d, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        sob = np.where(sob>np.max(sob)/2, 1, 0)
        res.append(np.sum(sob))
    res = res / np.max(res)
    return res


def sumColumnLine(X):
    res = []
    for img in range(len(X)):
        img_2d = np.reshape(X[img], (100, 100))
        sum_line = np.sum(img_2d, axis = 0)
        sum_column = np.sum(img_2d, axis = 1)
        res.append([*sum_line,*sum_column])
    return np.array(res)/100

  
def createNewFeatures(X):
    var = X.var(axis=1)
    std = X.std(axis=1)
    sob = calcPerimeter(X)
    X = binariseImage(X)
    X = sumColumnLine(X)
    X = np.c_[X,X.mean(axis=1)]
    X = np.c_[X,std]
    X = np.c_[X,var]
    X = np.c_[X,sob]
    return X

def max_indice(x):
        maxIndice = 0
        for i in range(len(x)):
            if x[i]>x[maxIndice]:
                maxIndice = i
        return maxIndice

def findBestSkb(X,Y, borne_inf = 1, borne_sup = 203, pas = 25):
    score_max = 0
    nb_features = 0

    if borne_inf < 0:
        borne_inf = 0
    if borne_sup > X.shape[1]:
        borne_sup = X.shape[1]

    for i in range(borne_inf,borne_sup,pas): #5998,6002,1
        Xsauv = np.copy(X)
        Ysauv = np.copy(Y)
        clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
        skb = SelectKBest(chi2, k = i)
        pipe = Pipeline([('skb', skb), ('clf', clf)])
        scoring_function1 = getattr(metrics, "balanced_accuracy_score")
        res = cross_val_score(pipe, Xsauv, Ysauv, cv=2 , scoring = make_scorer(scoring_function1))
        res = res.mean()
        if res > score_max:
            score_max = res
            nb_features = i

    if pas == 1 :
        print("max score for skb : ", score_max)
        return nb_features
    else :
        return findBestSkb(X,Y, borne_inf = nb_features-pas, borne_sup = nb_features + pas, pas = int(pas/2))


def findBestPca(X, Y, borne_inf = 1, borne_sup = 200, pas = 25, nb_feat = 200):
    score_max = 0
    nb_features = 0

    if borne_inf < 0:
        borne_inf = 0
    if borne_sup > X.shape[1]:
        borne_sup = X.shape[1]

    for i in range(borne_inf,borne_sup,pas):
        Xsauv = np.copy(X)
        Ysauv = np.copy(Y)
        clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
        skb = SelectKBest(chi2, k= nb_feat)
        pipe = Pipeline([('skb', skb), ('std', StandardScaler()), ('pca', KernelPCA(i)), ('clf', clf)])
        scoring_function1 = getattr(metrics, "balanced_accuracy_score")
        res = cross_val_score(pipe, Xsauv, Ysauv, cv=2 , scoring = make_scorer(scoring_function1))
        res = res.mean()
        if res > score_max:
            score_max = res
            nb_features = i
    if pas == 1 :
        print("max score for pca : ", score_max)
        return nb_features
    else :
        return findBestPca(X,Y, borne_inf = nb_features - pas, borne_sup = nb_features + pas, pas = int(pas/2))



def findBestKneighbors(X, Y, borne_inf = 1, borne_sup = 203, pas = 25):
    score_max = 0
    nb_features = 0

    if borne_inf < 0:
        borne_inf = 0

    if borne_sup > X.shape[1]:
        borne_sup = X.shape[1]

    for i in range(141,156,1):
        Xsauv = np.copy(X)
        Ysauv = np.copy(Y)
        Xsauv, Ysauv = Preprocessor.outliersDeletion(Xsauv, Ysauv,nbNeighbors=i)
        clf = RandomForestClassifier(n_estimators=196, max_depth=None, min_samples_split=2, random_state=1)
        scoring_function1 = getattr(metrics, "balanced_accuracy_score")
        res = cross_val_score(clf, Xsauv, Ysauv, cv=2 , scoring = make_scorer(scoring_function1))
        res = res.mean()
        if res > score_max:
            score_max = res
            nb_features = i

    if pas == 1 :
        print("max score for pca : ", score_max)
        return nb_features
    else :
        return findBestKneighbors(X,Y, borne_inf = nb_features-pas, borne_sup = nb_features + pas, pas = int(pas/2))
