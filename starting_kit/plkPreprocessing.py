"""
Created on Sat Mar 27 2020
Last revised: Mar 27, 2020
@author: mussard_romain
This program preprocessed data for classification.
Actually this program handle pca, features selection and outliers deletion

TODO :
    construct features (hard)
    include outliersDeletion on fit and fit_transform ?
    
préfèrer l'utilisation de fit_transform à fit
"""

model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir); 
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from data_manager import DataManager
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer

class Preprocessor():
    def __init__(self,n_components = 8, nb_feat = 193, nbNeighbors = 3):
        self.skb = SelectKBest(chi2, k= nb_feat)
        self.pca = PCA(n_components)
        self.lof = LocalOutlierFactor(n_neighbors=nbNeighbors)

    def fit(self, X, Y):
        '''
            Run score function on (X, Y) and get the appropriate features.
            
            Paramaters
            ----------
            X : array-like of shape (n_samples, n_features) representing The training input samples.
            Y : array-like of shape (n_samples,) representing The target values
            
            Returns
            -------
            self : object
            
        '''
        self.skb = self.skb.fit(X,Y)
        X_temp = self.skb.transform(X)
        self.pca = self.pca.fit(X_temp)
        return self

    def fit_transform(self, X, Y):
        '''
            Fit to data, then transform it.
            Fits transformer to X and Y with optional parameters fit_params 
            and returns a transformed version of X.
            
            Paramaters
            ----------
            X : array-like of shape (n_samples, n_features) representing the training input samples.
            Y : array-like of shape (n_samples,) representing The target values 
            
            Returns
            -------
            
            X_new : numpy array of shape [n_samples, n_features_new] representing the transformed array.
        '''
        return self.fit(X, Y).transform(X)

    def transform(self, X):
        '''
            Transform the data and returns a transformed version of X.
            
            Paramaters
            ----------
            X : array-like of shape (n_samples, n_features) representing the training input samples.
            
            Returns
            -------
            
            X_new : numpy array of shape [n_samples, n_features_new] representing the transformed array.
        '''
        X_res = self.skb.transform(X)
        X_res = self.pca.transform(X_res)
        return X_res
    
    def outliersDeletion(self, X, Y):
        '''
            Detect the outliers and return the training input samples (X)
            and the target values (Y) without the outliers
            
            Paramaters
            ----------
            X : array-like of shape (n_samples, n_features) representing the training input samples.
            Y : array-like of shape (n_samples,) representing The target values 
            
            Returns
            -------
            
            X_new : numpy array of shape [n_samples, n_features_new] representing the samples without outliers.
            Y_new : numpy array of shape [n_samples] representing the labels without ouliers.

        '''
        decision = self.lof.fit_predict(X)
        return X[(decision==1)],Y[(decision==1)]

if __name__=="__main__":
    data_dir = 'public_data'
    data_name = 'plankton'
    
    Prepro = Preprocessor()
    
    '''Show the original data before treatment'''
    D = DataManager(data_name, data_dir) # Load data
    print("*** Original data ***")
    print(D)
    
    '''Preproessing the data and show them after treatment'''
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
    D.feat_name = np.array(['PC1', 'PC2'])
    D.feat_type = np.array(['Numeric', 'Numeric'])
    print("*** Transformed data ***")
    print(D)
    
    '''Show data after Outliers deletion'''
    D.data['X_train'], D.data['Y_train'] = Prepro.outliersDeletion(D.data['X_train'],D.data['Y_train'])
    print("***Outliers Deletion***")
    print(D)
    tab = np.zeros(7)
    for i in D.data['Y_train']:
        tab[int(i)-1]+=1
    print(tab)
    
