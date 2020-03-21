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

def selectFeatures(data_target, nb_features, col = 'target'):
    '''Take a DataFrame et the number of features we want to keep and return an X and Y vectors
    representing the data and the labels'''
    corr = data_target.corr()
    col = data_target.columns
    print(corr['target'])
    sval = corr['target'][:-1].abs().sort_values(ascending=False)
    ranked_columns = sval.index.values
    col_selected = ranked_columns[0:nb_features]
    df = pd.DataFrame.copy(data_target)
    df = df[col_selected]
    df['target'] = data_target['target']
    X,Y = ExtractData(df)
    return X,Y,col_selected


def dimensionReduction(X_train, fnum):
    '''Make a dimension reduction on our data using pca
    X_train is our data
    fnum is the number of features we want
    return new data and pca'''
    labels= ['SV'+str(i) for i in range(1,fnum+1)]
    labels.append("target")
    pca = PCA(n_components=fnum)
    X = pca.fit_transform(X_train)
    return X, pca

def outliersIQR(X_train, Y_train, StDeviation):
    '''Erase outliers with the IQR
    X_train is our data
    Y_train is our label
    stDeviation is the coeff of multiplicaiton of the IQR
    we return new data and label'''
    XY = pd.DataFrame(np.append(X_train, Y_train, axis=1))
    Q1 = XY.quantile(0.15)
    Q3 = XY.quantile(0.75)
    IQR = Q3 - Q1
    XY = XY[~((XY < (Q1 - StDeviation * IQR)) |(XY > (Q3 + StDeviation * IQR))).any(axis=1)]
    X,Y = ExtractData(XY)
    print("Number of Data deleted = ", X_train.shape[0]-X.shape[0])
    return X,Y


def LocalOutFact(X_train, Y_train, nbNeighbors=35):
    '''Erase outliers with LocalOutlierFactor
    X_train is our data
    Y_train is our label
    nbNeighbors is a meta-parameter
    we return new data and label'''
    XY = pd.DataFrame(np.append(X_train, Y_train, axis=1))
    lof = LocalOutlierFactor(n_neighbors=nbNeighbors)
    decision = lof.fit_predict(X_train)
    XY = XY[~(decision==-1)]
    X,Y = ExtractData(XY)
    print("Number of Data deleted = ", X_train.shape[0]-X.shape[0])
    return X,Y

if __name__=="__main__":
    X = [[0.5,0.75,0.5],[1,-1,2], [0.25, 0.5,0.25]]
    Y = [[1],[-1],[1]]
    
    #test unitaire de maketrainAndTestData
    X_tr, Y_tr, X_test, Y_test = makeTrainAndTestData(X,Y)
    #"devrais afficher [[0.5, 0.75, 0.5], [1, -1, 2]]"
    print(X_tr)
    #"devrais afficher [[1],[-1]] \n"
    print(Y_tr)
    #"devrais afficher [[0.25, 0.5, 0.25]]"
    print(X_test)
    #"devrais afficher [1]"
    print(Y_test)
    
    
    #test unitaire selectfeatures()
    
    X = [[0.5,0.75,1],[1,-1,2], [0.25, 0.5,0.5]]
    Y = [[1],[-1],[1]]
    XY = pd.DataFrame(np.append(X, Y, axis=1), columns = ['a', 'b', 'c', 'target'])
    X,Y,col_sel = selectFeatures(XY,2, 'd')
    #colone b = 2*a donc forte coorelation entre a et b donc a devrais être supprimé 
    print("\n dataframe before : \n", XY, "\n")
    print("colonne gardé : ", col_sel, "\n")
    col_sel = list(col_sel)
    col_sel.append('target')
    XY = pd.DataFrame(np.append(X, Y, axis=1), columns = col_sel)
    print("dataframe après : \n", XY, "\n")

    #test unitaire PCA
    X = [[0.5,0.75,1,5],[1,-1,2,6], [0.25, 0.5,0.5,6], [0.25,-1,1,6]]
    Y = [[1],[-1],[1]]
    X_new, pca = dimensionReduction(X, 4)
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    #code ispired by this article https://gmaclenn.github.io/articles/airport-pca-analysis/
    plt.bar(list(range(1,len(evr)+1)), evr, alpha=0.75, align='center',
            label='individual explained variance')
    plt.step(list(range(1,len(evr)+1)), cum_evr, where='mid',
             label='cumulative explained variance')
    plt.ylim(0, 1.1)
    plt.xlim(0.5,4)
    plt.xticks(np.arange(1, 4, 1)) 
    plt.yticks(np.arange(0.0, 1.1, 0.1)) 
    plt.xlabel('Principal components')
    plt.ylabel('Explained variance ratio')
    plt.legend(loc='best')
    plt.show()
    
    #test unitaire outliers IQR
    
    #test unitaire outliers Kneigh
    
    