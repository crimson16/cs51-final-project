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
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import Initialize
import Accuracy
import Distance

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    # Depending on whether the training or testing dataset is needed,
    # read in the appropriate images and labels.
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    # If improper argument is provided to the "dataset" parameter,
    # raise an error.
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    # Read in labels file.
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()
    # Read in pixel values file.
    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    # Find indices of images whose labels are in the specified digit labels.
    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
    # Generate images and labels.
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels


######################################
# Load in training images and labels #
######################################
# load training images and labels as 60,000 x 28 x 28 array
train_images,train_labels = load_mnist("testing",path=os.getcwd())
# flattens train_labels from format [[3],[2],[6],...] to [3,2,6,...]
train_labels = [label[0] for label in train_labels]
# flatten training images into 60,000 x 784 array
train_images_flat = np.array([np.ravel(img) for img in train_images])

#########################
# Set parameter values  #
#########################
k = int(sys.argv[1]) # number of clusters (system argument)
n = len(train_images) # number of data points
l2 = len(train_images[0][0]) # number of rows in a training datapoint
l1 = len(train_images[0]) # number of columns in a single training datapoint
l = l1 * l2 # total number of pixels in a training datapoint
means = np.zeros((k,l)) # initial cluster means -- initialized to k x l 0s
                        # representing the pixels of the k cluster centers


#################################################
# Lloyd's algorithm -- find optimal clustering  #
#################################################

def kmeans(training_data, initial_clusters, distfn = Distance.sumsq, method = "means"):
  """
    Run the k-means (or k-medians, if the "medians" parameter is set
    to true) algorithm, based off of Lloyd's algorithm.
    
    Inputs
    -------
    training_data : Data off which to train the model. In the case of
    MNIST digits, it is a vector of pixel values in format 60,000 x 784

    initial_responsibilities : Vector of length n containing the initial
    cluster assignments of each of the datapoints. These can either be
    initialized randomly or by k-means++ and have format 60,000 x 784

    distfn : A function tdhat measures distance between points (i.e. sum of
    squared distances versus sum of absolute distances, etc.)

    method : Can be either "means","medoids", or "medians".

    Returns
    --------
    Vector of length n containing the final cluster assignments.
  """
  
  i = 0 # keep track of iteration

  r = np.zeros((n,k)) # create empty array to store cluster assignments

  # find and store k that minimize sum of square distance for each image
  newks = np.apply_along_axis(Distance.leastsquares, 1, training_data, initial_clusters, distfn)
  # create one hot coded vector for each image to signify cluster assignment
  r[range(n), newks] = 1

  # find new means
  while True:
    for smallk in range(k): # iterate through clusters
      ones = np.where(r[:,smallk]==1)[0]
      # The k-means method updates cluster centers as being the mean of each corresponding
      # pixel of the datapoints that are contained in that cluster.
      if method == "means":
        means[smallk,:] = np.mean(training_data[list(ones),:], axis=0)
      # The k-medoids method updates cluster centers as being the closest *datapoint*
      # to the mean of the corresponding pixel of datapoints contained in that cluster.
      elif method == "medoids":
        distance_point_to_center = np.sum((training_data[list(ones),:] 
          - np.mean(training_data[list(ones),:], axis=0))**2,axis=1)
        means[smallk,:] = training_data[list(ones),:][np.argmin(distance_point_to_center)]
      # The k-medians method updates cluster centeras as being the median of each corresponding
      # pixel of the datapoints that are contained in that cluster.
      elif method == "medians":
        means[smallk,:] = np.median(training_data[list(ones),:], axis=0)
      # If no proper value is chosen for method, then return error.
      else:
        raise ValueError("Not a valid method specification; must be 'means', 'medoids', or 'medians'")

    # update responsibilities by minimizing sum of squared distances
    r_new = np.zeros((n,k))

    # stores indices of k's that minimize ssd
    newks = np.apply_along_axis(Distance.leastsquares, 1, training_data, means, distfn)
    r_new[range(n), newks] = 1

    # if none of the responsibilities change, then we've reached the optimal cluster assignments
    if np.all((r_new - r)==0):
      return r, means
    else:
      r = r_new
    # After each iteration, print iteration number and the number of images assigned to a given cluster.
    print i, r.sum(axis=0)
    i += 1

  print 'finished'

final_responsibilities, final_clusters = kmeans(train_images_flat, Initialize.random_centers(k), 
  distfn = Distance.sumsq, method = "medoids")
print final_responsibilities.sum(axis=0)

# processes and saves mean images and randomly selected images from each cluster
'''
imgnum = 4
data = np.zeros((32, 32, 3), dtype=np.uint8)
cluster_data = np.zeros((32, 32, 3), dtype=np.uint8)
for j in range(K):  
  # saves the mean of each cluster as an image named 'K_meanj.png'
  data[:,:,0] = np.reshape(means[j, 0:1024], (32,32))
  data[:,:,1] = np.reshape(means[j, 1024:2048], (32,32))
  data[:,:,2] = np.reshape(means[j, 2048:3072], (32,32))
  img = Image.fromarray(data, 'RGB')
  img.save('./cluster' + str(K) + '/mean' + str(j) + '.png')

  # randomly saves imgnum images from each cluster
  indices = np.nonzero(r[:, j])
  sample = random.sample(indices[0], imgnum)
  counter = 0
  
  for n in sample:
    cluster_data[:,:,0] = np.reshape(x[n, 0:1024], (32,32))
    cluster_data[:,:,1] = np.reshape(x[n, 1024:2048], (32,32))
    cluster_data[:,:,2] = np.reshape(x[n, 2048:3072], (32,32))
    img = Image.fromarray(cluster_data, 'RGB')
    img.save('./cluster' + str(K) + '/cluster' + str(j) + '_' + str(counter) + '.png')
    counter += 1

'''

Accuracy.final_accuracy(final_responsibilities, train_labels, train_images_flat, final_clusters)