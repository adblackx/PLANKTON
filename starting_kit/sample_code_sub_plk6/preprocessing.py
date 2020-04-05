"""
Created on Sat Mar 27 2020
Last revised: Mar 27, 2020
@author: mussard_romain
This program preprocessed data for classification.
Actually this program handle pca, features selection and outliers deletion

We still have to :
    normalize data
    construct features
    
préfèrer l'utilisation de fit_transform à fit
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris


class Preprocessor(BaseEstimator):
    def __init__(self,n_components = 2, nb_feat = 193, nbNeighbors = 7):
        self.skb = SelectKBest(chi2, k= nb_feat)
        self.pca = PCA(n_components)
        self.lof = LocalOutlierFactor(n_neighbors=nbNeighbors)

    def fit(self, X, Y):
        print("nb_feat", nb_feat)
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
    
    def outliersDeletion(self, X, Y):
        decision = self.lof.fit_predict(X)
        return X[(decision==1)],Y[(decision==1)]
