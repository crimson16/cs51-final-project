####
# kmeans_raw.py
#
# Implements k-means clustering without k-means++.
# Uses MNIST handwritten digit training dataset.
# Takes in a value of k (# of clusters) as a system argument.
#
# Olivia Angiuli, Martin Reindl, Ty Rocca, Wilder Wohns
###

#################
# Load packages #
#################

import numpy as np
import os, struct,random,sys
from numpy import append, array, int8, uint8, zeros
import Initialize
import Accuracy
import Distance
import Load
import Kmeans
import Image

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

#################################################
# Lloyd's algorithm -- find optimal clustering  #
#################################################

# Run clustering algorithm
final_responsibilities, final_clusters = Kmeans.kmeans(k, train_images_flat,
  Initialize.random_centers(k), distfn = Distance.sumsq, method = "means")
print final_responsibilities.sum(axis=0)

# Save representative images to file.
Load.save_images(k, train_images, final_responsibilities, final_clusters)

# Calculate final accuracy
Accuracy.final_accuracy(final_responsibilities, train_labels, train_images_flat, final_clusters)