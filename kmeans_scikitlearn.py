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
import Accuracy

######################################
# Load in training images and labels #
######################################
# load training and testing images and labels as 60,000 x 28 x 28 array
train_images,train_labels = Load.load_mnist("training",path=os.getcwd())
test_images,test_labels = Load.load_mnist("testing",path=os.getcwd())
# flatten training images into 60,000 x 784 array
train_images_flat = np.array([np.ravel(img) for img in train_images])
test_images_flat = np.array([np.ravel(img) for img in test_images])

#########################
# Set parameter values  #
#########################
k = int(sys.argv[1]) # number of clusters (system argument)

# Train k means model
kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
kmeans_fit = kmeans.fit(train_images_flat)
# Get the cluster assignments of each point of training images
kmeans_labels = kmeans_fit.labels_
kmeans_centers = kmeans_fit.cluster_centers_ 

# Initialize a vector of responsibilities in a one-hot-coded format.
final_responsibilities = np.zeros((len(train_images_flat),k))
# For each cluster assignment, assign the appropriate vector in the
# one-hot-coded format to a 1.
for imgnum in range(len(train_images_flat)):
	final_responsibilities[imgnum][kmeans_labels[imgnum]] = 1


# Obtain predictions for each point.
Z = kmeans.predict(test_images_flat)

# Determine accuracies.
Accuracy.final_accuracy(final_responsibilities, train_labels, train_images_flat, kmeans_centers)