model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir); 
# Uncomment the next lines to auto-reload libraries (this causes some problem with pickles in Python 3)
import seaborn as sns; sns.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score as sklearn_metric
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_io import read_as_df


def ExtractData(df):
    '''Extract data and label from a dataFrame'''
    X = np.array(df.iloc[:,:-1])
    Y = np.array(df.iloc[:,-1])
    Y = Y.reshape(Y.shape[0],1)
    return X,Y

def makeTrainAndTestData(X_donnees, Y_donnees):
    '''Separate our data on two : a set of training and a set of validation'''
    decoupe = int(2*len(X_donnees)/3)
    X_train = X_donnees[0:decoupe]
    Y_train = Y_donnees[0:decoupe]
    
    X_test = X_donnees[decoupe:]
    Y_test = Y_donnees[decoupe:]
    
    return X_train, Y_train, X_test, Y_test 

def selectFeatures(data_target, nb_features):
    '''Take a DataFrame et the number of features we want to keep and return an X and Y vectors
    representing the data and the labels'''
    corr = data_target.corr()
    print(corr['target'])
    sval = corr['target'][:-1].abs().sort_values(ascending=False)
    ranked_columns = sval.index.values
    col_selected = ranked_columns[0:nb_features]
    df = pd.DataFrame.copy(data_target)
    df = df[col_selected]
    df['target'] = data_target['target']
    X,Y = ExtractData(df)
    return X,Y


def dimensionReduction(X_train, fnum):
    '''Make a dimension reduction on our data'''
    labels= ['SV'+str(i) for i in range(1,fnum+1)]
    labels.append("target")
    pca = PCA(n_components=fnum)
    X = pca.fit_transform(X_train)
    return X, pca

def outliersIQR(X_train, Y_train, StDeviation):
    '''Erase outliers with the IQR'''
    XY = pd.DataFrame(np.append(X_train, Y_train, axis=1))
    #orignaly used with 0.15 et 0.75 but that delete to much data
    Q1 = XY.quantile(0.15)
    Q3 = XY.quantile(0.75)
    IQR = Q3 - Q1
    XY = XY[~((XY < (Q1 - StDeviation * IQR)) |(XY > (Q3 + StDeviation * IQR))).any(axis=1)]
    X,Y = ExtractData(XY)
    print("Number of Data deleted = ", X_train.shape[0]-X.shape[0])
    return X,Y


def LocalOutFact(X_train, Y_train, nbNeighbors):
    '''Erase outliers with LocalOutlierFactor'''
    XY = pd.DataFrame(np.append(X_train, Y_train, axis=1))
    lof = LocalOutlierFactor(n_neighbors=nbNeighbors)
    decision = lof.fit_predict(X_train)
    XY = XY[~(decision==-1)]
    X,Y = ExtractData(XY)
    print("Number of Data deleted = ", X_train.shape[0]-X.shape[0])
    return X,Y

