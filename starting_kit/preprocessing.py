"""
Created on Sat Mar 27 2020
Last revised: Mar 27, 2020
@author: mussard_romain
This program preprocessed data for classification.
Actually this program handle pca, features selection and outliers deletion

We still have to :
    normalize data
    construct features
"""

from sys import path 
import pandas as pd
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from data_manager import DataManager
from sklearn.preprocessing import MinMaxScaler


class Preprocessor():
    def __init__(self,n_components = 2, nb_feat = 193, nbNeighbors = 7):
        self.skb = SelectKBest(chi2, k= nb_feat)
        self.pca = PCA(n_components)
        self.lof = LocalOutlierFactor(n_neighbors=nbNeighbors)

    def fit(self, X, Y):
        self.skb = self.skb.fit(X,Y)
        self.pca = self.pca.fit(X)
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

if __name__=="__main__":
    data_dir = 'public_data'
    data_name = 'plankton'
    
    Prepro = Preprocessor()
    
    D = DataManager(data_name, data_dir) # Load data
    print("*** Original data ***")
    print(D)
    
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
    D.feat_name = np.array(['PC1', 'PC2'])
    D.feat_type = np.array(['Numeric', 'Numeric'])
    print("*** Transformed data ***")
    print(D)
    
    D.data['X_train'], D.data['Y_train'] = Prepro.outliersDeletion(D.data['X_train'],D.data['Y_train'])
    print("***Outliers Deletion***")
    print(D)

    
    
