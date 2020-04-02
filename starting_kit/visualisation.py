#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:07:53 2020

@author: Carlo-Elia DONCECCHI & Ugo NZONGANI

This files contains the code produced by the Visualization binomial of the Plankton team.
Our goal is to create usefull visualization in order to have a better understanding of the
problem and a better interpretation of the results

Here is the link of our Jupyter Notebook:
https://github.com/adblackx/PLANKTON/blob/master/starting_kit/README-Visualisation.ipynb

Last update: 02/04/2020
"""
model_dir = 'sample_code_submission/'
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
data_dir = 'public_data'
data_name = 'plankton'
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
    from data_manager import DataManager
    from model import model
    from libscores import get_metric
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale
    from sklearn.tree import DecisionTreeClassifier
    # These imports are not used but we need them if we want to try the decision surface function with many classifiers
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
K-means function:
Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
This function is used to visualize the clusters in our data using the k-means algorithm
---------------------------------------------------------
Args
    X: Training data array
    y: Training label array
    marker_settings = [marker_type, marker_size, marker_color]
    colors: color of the point for each different class
    title: title of the figure
    title_color: title's color
    title_size: title's size
    title_weight: normal or bold
----------------------------------------------------------
"""

def kmeans(X, y, marker_settings, colors, title, title_color, title_size, title_weight):
    #np.random.seed(42)

    data = scale(X)
    # We recover the numbers of features and data (pictures of different plankton)
    n_data, n_features = data.shape
    
    # We recover the number of classes
    n_classes = len(np.unique(y))
    print("n_classes: %d, \t n_data %d, \t n_features %d"% (n_classes, n_data, n_features))

    # We reduce the data dimension to 2 instead of 203 using PCA
    reduced_data = PCA(n_components=2).fit_transform(data)
    
    # We initialize the k-means
    kmeans = KMeans(init='k-means++', n_clusters=n_classes, n_init=7)
    
    # We train the k-means
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # We recover the predicted label for each point
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
    # Plot the points: the argument 'k.' is used to plot the points with a shape of 'point', and not square for example
    for i, color in zip(range(n_classes), colors):
        idx = np.where(y == i)
        plt.plot(reduced_data[idx, 0], reduced_data[idx, 1], 'k.', markersize=2,color=color)
    # Plot the centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker=marker_settings[0], s=marker_settings[1], linewidths=3, color=marker_settings[2], zorder=10)
    # We add the title and its settings
    plt.title(title, color=title_color, fontsize = title_size, fontweight=title_weight)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
"""
Decision surface function:
Source: https://scikit-learn.org/0.15/auto_examples/tree/plot_iris.html
This function is used to plot the decision surface of a classifier, the decision surface is a good way
to see the efficiancy of a classifier
------------------------------------------------------------------
Args
    X: Training data array
    y: Training label array
    colors: Color of the point for each different class
    classes_names: Array which contains the names of the classes
    Classifier: The classifier used
    title: Title of the figure
    title_color: Title's color
    title_size: Title's size
    title_weight: Normal or bold
-------------------------------------------------------------------
"""

def decision_surface(X, y, colors, classes_names, Classifier, title, title_color, title_size, title_weight):
    
    #We recover the number of classes
    classes_number = len(np.unique(y))
    
    # We reduce the data dimension to 2 instead of 203 using PCA
    X = PCA(n_components=2).fit_transform(scale(X))
    reduced_data = X

    # We train our classifier
    clf = Classifier().fit(reduced_data, y)
    
    # Plot the decision boundary
    plt.subplot()
    plot_step = 0.02
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

    # We recover the predicted label by the classifier
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the background's colors
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
   
    # Plot the training points with the color of its class
    for i, color in zip(range(classes_number), colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=classes_names[i], s=3, cmap=plt.cm.Paired) 
    
    # We add the title and its settings
    plt.axis("tight")
    plt.suptitle(title, color=title_color, fontsize=title_size, fontweight=title_weight)
    plt.legend(title='Classes',bbox_to_anchor=(1.2,0.5,0.25,0.),loc=5)
    plt.show()
    

"""
Cross-validation plot function:
This function is used to plot the cross-validation's graph
---------------------------------------------------------------
Args
    score: Array which contains the cross-validation score
    settings: [curve_color, marker_type, marker_size, marker_color]
    title: Title of the figure
    title_color: Title's color
    title_size: Title's size
    title_weight: Normal or bold
-----------------------------------------------------------------
"""

def plot_cross_validation(scores, settings, title, title_color, title_size, title_weight):
    # Plot the figure
    plt.plot(scores, color=settings[0], marker=settings[1], markersize=settings[2], MarkerFaceColor=settings[3],
             MarkerEdgeColor="Black")
    # We add the name of the axis
    plt.ylabel('Score')
    plt.xlabel('(x-ieme +1) cross validation')
    # We add the title and it's settings
    plt.title('Cross validation performance', color=title_color, fontsize = title_size, fontweight=title_weight)
    plt.show()

"""
Model's performance function:
This function is used to obtain the model's performance
Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
---------------------------------------------------------------
Args
    X: Training data array
    y: Training label array
    training_size: Array which contains the different values of the training set size
    M: model
    scoring_function: metric
---------------------------------------------------------------
"""

def model_performance(X, y, training_size, M, scoring_function):
    train_sizes, train_scores, test_scores = learning_curve(M, X, y, train_sizes = training_size,
                                                        cv=5, scoring=make_scorer(scoring_function))
    score = np.mean(test_scores, axis=1)
    score_error = np.std(test_scores, axis=1)
    res = [score, score_error]
    return res

"""
Plot performance function:
This function is used to plot the model's performance with errors bars
-----------------------------------------------------------------------
Args
    training_size: Array which contains the different values of the training set size, it has to be the same as
                    you used for the function model_performance
    performance: values of y axis, it correspond to the result of the function model_performance
    settings: [curve_color, marker_type, marker_size, marker_color]
    title: Title of the figure
    title_color: Title's color
    title_size: Title's size
    title_weight: Normal or bold
-----------------------------------------------------------------------
"""

def plot_performance(training_size, performance, settings, title, title_color, title_size, title_weight):
    # Plot the model's performance with errors bars
    plt.errorbar(training_size, performance[0], performance[1], color=settings[0], marker=settings[1],
                 markersize=settings[2], MarkerFaceColor=settings[3], MarkerEdgeColor="Black")
    # We add the name of the axis
    plt.xlabel("Training set size")
    plt.ylabel("Performance (score)")
    # We add the title and it's settings
    plt.title(title, color=title_color, fontsize=title_size, fontweight=title_weight)
    plt.show()


"""
Main function
"""

if __name__=="__main__":
    # We load the data
    D = DataManager(data_name, data_dir, replace_missing=True)
    X = D.data['X_train']
    y = D.data['Y_train']
    M = model(classifier=DecisionTreeClassifier(max_depth=10, max_features = 'sqrt',random_state=42))
    metric_name, scoring_function = get_metric()
    scores = cross_val_score(M, X, y, cv=5, scoring=make_scorer(scoring_function))
    
    # K-means function
    settings = ['x', 150, 'w']
    kmeans(X, y, settings, "ymkrgbc",
       'K-means clustering on the digits dataset (PCA-reduced data)', 'Red', 13, 'normal')
    
    # Decision surface function
    classes_names = ["chaetognatha","copepoda","euphasiids","fish_larvae","limacina","medusae","other"]

    decision_surface(X, y, "mykrgbc", classes_names, DecisionTreeClassifier, 'Decision surface of a DecisionTreeClassifier', 'Green', 12, 'bold')
    
    # Cross-validation plot function
    settings = ['Purple', 's', 7, 'Pink']
    plot_cross_validation(scores, settings, 'Cross validation performance', 'Blue', 12, 'bold')
    
    # First we use the model_performance function to obtain the model's performance
    t = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 8601]
    perf = model_performance(X, y, t, M, scoring_function)
    
    # Now we plot the performance using the previous results
    settings = ['Red', '8', 6, 'Blue']
    plot_performance(t, perf, settings, "Machine learning model's performance according to the size of the training set",
                 'Brown', 11, 'bold')
    