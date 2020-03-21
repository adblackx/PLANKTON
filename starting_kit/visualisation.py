#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:07:53 2020

@author: ugo
"""
model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
from sys import path; 
path.append(model_dir); 
path.append(problem_dir); 
path.append(score_dir); 

import warnings

with warnings.catch_warnings():
	# Uncomment the next lines to auto-reload libraries (this causes some problem with pickles in Python 3)
    import seaborn as sns; sns.set()
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import matplotlib.pyplot as plt
    import numpy as np
    data_dir = 'public_data'          # The sample_data directory should contain only a very small subset of the data
    data_name = 'plankton'
    from data_manager import DataManager
    from model import model
    from libscores import get_metric
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import learning_curve

"""
Fonction pour les k-moyennes:
Source:Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
#X: ensemble d'apprentissage; y: labels
#param = paramètre des marker = [marker, size, color]
#color: couleur des points de chaque classe
"""

def kmeans(X, y, param, color, title, title_color, title_size, title_weight):
    np.random.seed(42)

    X_digits = X
    y_digits = y
    data = scale(X_digits)
    plot_colors = color
    n_samples, n_features = data.shape
    n_digits = len(np.unique(y_digits))
    print("n_digits: %d, \t n_samples %d, \t n_features %d"% (n_digits, n_samples, n_features))

    # #############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=7)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
    for i, color in zip(range(n_digits), plot_colors):
        idx = np.where(y == i)
        plt.plot(reduced_data[idx, 0], reduced_data[idx, 1], 'k.', markersize=2,color=color)
    # Plot the centroids as a red circle
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker=param[0], s=param[1], linewidths=3, color=param[2], zorder=10)
    plt.title(title, color=title_color, fontsize = title_size, fontweight=title_weight)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
"""
Fonction pour la surface de décision:
Source:https://scikit-learn.org/0.15/auto_examples/tree/plot_iris.html
#X: ensemble d'apprentissage; y: labels
#nb_classes: nombre de classe
#classes_names: nom des classes
#Classifier: classifieur utilisé
"""
def decision_surface(X, y, nb_classes, colors, classes_names, Classifier, title, title_color, title_size, 
                    title_weight):
    # Parameters
    n_classes = 7
    plot_colors = colors
    plot_step = 0.02

    data = scale(X)
    n_digits = len(np.unique(y))
    # Visualize the results on PCA-reduced data
    X = PCA(n_components=2).fit_transform(data)

    # Train
    clf = Classifier().fit(X, y)
    
    # Plot the decision boundary
    plt.subplot()
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis("tight")
    
    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=classes_names[i], s=3, cmap=plt.cm.Paired)

    plt.axis("tight")
    plt.suptitle(title, color=title_color, fontsize=title_size, fontweight=title_weight)
    plt.legend(title='Classes',bbox_to_anchor=(1.2,0.5,0.25,0.),loc=5)
    plt.show()
    

"""
Fonction pour afficher le score de la cross-validation
#scores: score obtenu
#p = paramètres = [curve_color, marker_type, marker_size, marker_color]
"""
def plot_cross_validation(scores, p, title, title_color, title_size, title_weight):
    plt.plot(scores, color=p[0], marker=p[1], markersize=p[2], MarkerFaceColor=p[3], MarkerEdgeColor="Black")
    plt.ylabel('Score')
    plt.xlabel('(x-ieme +1) cross validation')
    plt.title('Cross validation performance', color=title_color, fontsize = title_size, fontweight=title_weight)

"""
Fonction pour obtenir la performance du modèle nécessaire pour la fonction plot_performance
Source:https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#X: ensemble d'apprentissage; y: labels
#t: tableau contenant les valeurs de la taille de l'ensemble d'apprentissage
#M: modèle
"""
def model_performance(X, y, t, M, scoring_function):
    train_sizes, train_scores, test_scores = learning_curve(M, X, y, train_sizes = t,
                                                        cv=5, scoring=make_scorer(scoring_function))
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    res = [test_scores_mean, test_scores_std]
    return res

"""
Fonction pour afficher le graphe des performances du modèle en fonction de la taille de l'ensemble d'apprentissage
avec des barres d'erreurs:
#t: valeurs des abscisses
#res: valeur des ordonnées, correspondant au résultat de la fonction model_performance
#p = paramètres = [curve_color, marker_type, marker_size, marker_color]
"""
def plot_performance(t, res, p, title, title_color, title_size, title_weight):
    plt.errorbar(t, res[0], res[1], color=p[0], marker=p[1], markersize=p[2], MarkerFaceColor=p[3], MarkerEdgeColor="Black")
    plt.xlabel("Training set size")
    plt.ylabel("Performance (score)")
    plt.title(title, color=title_color, fontsize=title_size, fontweight=title_weight)



if __name__=="__main__":
    D = DataManager(data_name, data_dir, replace_missing=True) # We reload the data with the AutoML DataManager class because this is more convenient
    X = D.data['X_train']
    y = D.data['Y_train']
    M = model(classifier=DecisionTreeClassifier(max_depth=10, max_features = 'sqrt',random_state=42))
    metric_name, scoring_function = get_metric()
    scores = cross_val_score(M, X, y, cv=5, scoring=make_scorer(scoring_function))
    
    "k-moyennes:"
    param = ['x', 150, 'w']
    kmeans(X, y, param, "ymkrgbc",
       'K-means clustering on the digits dataset (PCA-reduced data)', 'Pink', 13, 'bold')
    
    "decision surface:"
    classes_names = ["chaetognatha","copepoda","euphasiids","fish_larvae","limacina","medusae","other"]

    decision_surface(X, y, 7, "mykrgbc", classes_names, DecisionTreeClassifier,
                 'Decision surface of a DecisionTreeClassifier', 'Green', 12, 'bold')
    
    "cross-validation:"
    p = ['Purple', 's', 7, 'Pink']
    plot_cross_validation(scores, p, 'Cross validation performance', 'Blue', 12, 'bold')
    
    "performance:"
    t = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 8601]
    res = model_performance(X, y, t, M, scoring_function)
    param=['Yellow', '8', 6, 'Blue']
    plot_performance(t, res, param, "Machine learning model's performance according to the size of the training set",
                 'Brown', 11, 'bold')