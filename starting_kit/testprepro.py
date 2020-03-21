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
import preprocessing as prepro
#%% cellule 1
#data_dir = 'sample_data'              # Change this to the directory where you put the input data
data_dir = 'public_data'          # The sample_data directory should contain only a very small subset of the data
data_name = 'plankton'
data = read_as_df(data_dir  + '/' + data_name)                # The data are loaded as a Pandas Data Frame
data['target'].value_counts()
print(data.head())

#%% cellule 2 : Hsitogramme
data.describe()
data[['sum_axis_0_50','sum_axis_1_50','mean','variance', 'outline_length']].hist(figsize=(10, 10), bins=50, layout=(3, 2)) 

#%% cellule 3 : Matrice de Correlation
data_target = data.copy()
le = preprocessing.LabelEncoder()
data_target.target = le.fit_transform(data_target.target.values)
fig = plt.figure(figsize=(15,8))
sns.heatmap(data_target[['sum_axis_0_50','sum_axis_1_50','mean','variance','outline_length','target']].corr(), annot = True)
plt.title('Correlation_matrix')
plt.show()

#%% cellule 4
sns.pairplot(data,hue='target',vars=['sum_axis_0_50','sum_axis_1_50','mean','variance','outline_length'])
plt.show()

#å%% celule 5 : Feature Selection
clf = DecisionTreeClassifier(max_depth=10, max_features = 'sqrt',random_state=42)

score = []
nbFeatures = []
nbInitialFeatures = data.columns.size - 1

for i in range(203,0, -10):
    data_fs = data.copy()
    le = preprocessing.LabelEncoder()
    data_fs.target = le.fit_transform(data_fs.target.values)
    X_fs,Y_fs = prepro.selectFeatures(data_fs, i)
    X_train, Y_train, X_test, Y_test = prepro.makeTrainAndTestData(X_fs, Y_fs)
    clf.fit(X_train, Y_train)
    Y_predict =  clf.predict(X_test)
    score.append(sklearn_metric(Y_test, Y_predict))
    nbFeatures.append(i) 

x = np.array(nbFeatures)
y = np.array(score)
fig,ax = plt.subplots()
ax.plot(x,y,label = "Score d'apprentissage")

plt.title('Score according to number of features')
plt.xlabel('Number of features')
plt.ylabel('score')
plt.legend()

#%% celule 6 : PCA

data_fs = data.copy()
le = preprocessing.LabelEncoder()
data_fs.target = le.fit_transform(data_fs.target.values)
X_fs,Y_fs = prepro.ExtractData(data_fs)
X_fs, pca = prepro.dimensionReduction(X_fs, 13)
ev = pca.explained_variance_
evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)
#code ispired by this article https://gmaclenn.github.io/articles/airport-pca-analysis/
plt.bar(list(range(1,len(evr)+1)), evr, alpha=0.75, align='center',
        label='individual explained variance')
plt.step(list(range(1,len(evr)+1)), cum_evr, where='mid',
         label='cumulative explained variance')
plt.ylim(0, 1.1)
plt.xlim(0.5,14)
plt.xticks(np.arange(1, 14, 1)) 
plt.yticks(np.arange(0.0, 1.1, 0.1)) 
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')
plt.show()

#%% celule 7: Outlierd detection IQR
data_fs = data.copy()
le = preprocessing.LabelEncoder()
data_fs.target = le.fit_transform(data_fs.target.values)

X_fs,Y_fs = prepro.ExtractData(data_fs)
scaler = StandardScaler()
X_fs = scaler.fit_transform(X_fs)
X_fs, pca = prepro.dimensionReduction(X_fs, 2)

XY = pd.DataFrame(np.append(X_fs, Y_fs, axis=1))
#orignaly used with 0.15 et 0.75 but that delete to much data
Q1 = XY.quantile(0.15)
Q3 = XY.quantile(0.75)
IQR = Q3 - Q1
decision = ((XY > (Q1 - 1.5 * IQR))&(XY < (Q3 + 1.5 * IQR))).all(axis=1)

X_out = X_fs[~(decision==1)]
X_in = X_fs[~(decision==-1)]
colors = np.array(['#377eb8', '#ff7f00']) #377eb8 = bleu et #ff7ff0 = orange
plt.scatter(X_in[:, 0], X_in[:, 1],color = '#377eb8' , s=10, label= "Data")
plt.scatter(X_out[:, 0], X_out[:, 1],color = '#ff7f00' , s=10, label= "Outliers")
plt.legend()

#%% cellule 8:Outliers detection LocalOutFac
  
data_fs = data.copy()
le = preprocessing.LabelEncoder()
data_fs.target = le.fit_transform(data_fs.target.values)
X_fs,Y_fs = prepro.ExtractData(data_fs)
scaler = StandardScaler()
X_fs = scaler.fit_transform(X_fs)
X_fs, pca = prepro.dimensionReduction(X_fs, 2)
lof = LocalOutlierFactor(n_neighbors=35)
decision = lof.fit_predict(X_fs)
X_out = X_fs[~(decision==1)]
X_in = X_fs[~(decision==-1)]
colors = np.array(['#377eb8', '#ff7f00']) #377eb8 = bleu et #ff7ff0 = orange
plt.scatter(X_in[:, 0], X_in[:, 1],color = '#377eb8' , s=10, label= "Data")
plt.scatter(X_out[:, 0], X_out[:, 1],color = '#ff7f00' , s=10, label= "Outliers")
plt.legend()