####
# kmeans_scikitlearn.py
#
# Implements k-means clustering using scikit_learn.
# Meant to act as a comparison method to our raw implementation
# of k-means clustering.
#
# Olivia Angiuli, Martin Reindl, Ty Rocca, Wilder Wohns
###


from time import time
import os, struct,random,sys
import numpy as np
import matplotlib.pyplot as plt

from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

from sklearn.cluster import KMeans

import Load

######################################
# Load in training images and labels #
######################################
# load training and testing images and labels as 60,000 x 28 x 28 array
train_images,train_labels = Load.load_mnist("training",path=os.getcwd())
test_images,test_labels = Load.load_mnist("testing",path=os.getcwd())
# flattens train_labels from format [[3],[2],[6],...] to [3,2,6,...]
train_labels = [label[0] for label in train_labels]
test_labels = [label[0] for label in test_labels]
# flatten training images into 60,000 x 784 array
train_images_flat = np.array([np.ravel(img) for img in train_images])
test_images_flat = np.array([np.ravel(img) for img in test_images])

#########################
# Set parameter values  #
#########################
k = int(sys.argv[1]) # number of clusters (system argument)
n = len(train_images) # number of data points
l2 = len(train_images[0][0]) # number of rows in a training datapoint (assumes each training datapoint is same size)
l1 = len(train_images[0]) # number of columns in a single training datapoint (assumes each training datapoint is same size)
l = l1 * l2 # total number of pixels in a training datapoint
r = np.zeros((n,k)) # matrix of responsibilities (assignments of each datapoint to a cluster)
means = np.zeros((k,l))

# Train k means model
kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
kmeans_fit = kmeans.fit(train_images_flat)
kmeans_labels = kmeans_fit.labels_
print kmeans_labels


# Given the true labels of the training images (train_labels), determine
# the majority element (and therefore the labelling of each of the clusters)
cluster_labels = []
for assigned_cluster in range(k):
    cluster_points = np.where(kmeans_labels==assigned_cluster)[0]
    # labels of training points that were assigned to assigned_cluster
    cluster_true_labels = train_labels[cluster_points]
    cluster_true_labels = [int(label[0]) for label in cluster_true_labels]
    # vector containing the true "label" of each of the k clusters
    cluster_labels.append(np.argmax(np.bincount(cluster_true_labels)))


# Obtain predictions for each point.
Z = kmeans.predict(test_images_flat)

# For a given cluster, take its "labelling" in order to be whatever
# true class the majority of the training elements belong to.
#k
# Use this method in order to assign an accuracy score (1 or 0)
# based on whether the training data was accurately classified.

cluster_assignment_accuracy = [1 if test_labels[j]==cluster_labels[Z[j]] else 0 for j in range(len(test_labels))]
print float(np.sum(cluster_assignment_accuracy))/len(cluster_assignment_accuracy)