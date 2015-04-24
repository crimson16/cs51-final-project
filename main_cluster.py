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
l2 = len(train_images[0][0]) # number of rows in a training datapoint
l1 = len(train_images[0]) # number of columns in a single training datapoint
l = l1 * l2 # total number of pixels in a training datapoint

#################################################
# Lloyd's algorithm -- find optimal clustering  #
#################################################

final_responsibilities, final_clusters = Kmeans.kmeans(k, train_images_flat,
  Initialize.random_centers(k), distfn = Distance.sumsq, method = "medoids")
print final_responsibilities.sum(axis=0)
print final_clusters

# processes and saves mean images and randomly selected images from each cluster

imgnum = 4
data = np.zeros((28, 28, 3), dtype=np.uint8)
cluster_data = np.zeros((28, 28, 3), dtype=np.uint8)
for j in range(k):  
  # saves the mean of each cluster as an image named 'K_meanj.png'
  data[:,:,0] = np.reshape(final_clusters[j, :], (28,28))
  data[:,:,1] = np.reshape(final_clusters[j, :], (28,28))
  data[:,:,2] = np.reshape(final_clusters[j, :], (28,28))
  img = Image.fromarray(data, 'RGB')
  img.save('./cluster' + str(k) + '/mean' + str(j) + '.png')

  # randomly saves imgnum images from each cluster
  indices = np.nonzero(r[:, j])
  sample = random.sample(indices[0], imgnum)
  counter = 0
  
  for sampled_n in sample:
    cluster_data[:,:,0] = np.reshape(x[sampled_n, 0:1024], (28,28))
    cluster_data[:,:,1] = np.reshape(x[sampled_n, 1024:2048], (28,28))
    cluster_data[:,:,2] = np.reshape(x[sampled_n, 2048:3072], (28,28))
    img = Image.fromarray(cluster_data, 'RGB')
    img.save('./cluster' + str(k) + '/cluster' + str(j) + '_' + str(counter) + '.png')
    counter += 1



Accuracy.final_accuracy(final_responsibilities, train_labels, train_images_flat, final_clusters)