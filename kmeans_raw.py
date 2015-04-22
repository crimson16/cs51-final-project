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

global error

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()


    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

##########################
## Function definitions ##
##########################


# iterates over means matrix to calculate sum of distances
def sum(m, xn):
    diff = xn - m
    return diff.sum(axis=0)

# iterates over means matrix to calculate sum of absolute distances
def abs_sum(m, xn):
    diff = abs(xn - m)
    return diff.sum(axis=0)

# iterates over means matrix to calculate sum of squared distances
def sumsq(m, xn):
    diff = (xn - m) ** 2
    return diff.sum(axis=0)

# iterates over means matrix to calculate maximum distances
def maxdist(m, xn):
    diff = (xn - m) ** 2
    return diff.max(axis=0)

# returns index of cluster that minimizes error
def leastsquares(xn, means, dist_fn):
    error = 0
    errors = np.apply_along_axis(dist_fn, 1, means, xn) 
    error += errors[np.argmin(errors)]
    return np.argmin(errors)


######################################
# Load in training images and labels #
######################################
# load training images and labels as 60,000 x 28 x 28 array
train_images,train_labels = load_mnist("training",path=os.getcwd())
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
means = np.zeros((k,l))

#################################################
# Lloyd's algorithm -- find optimal clustering  #
#################################################

def kmeans(training_data, initial_clusters, distfn = sumsq, method = "means"):
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

    distfn : A function that measures distance between points (i.e. sum of
    squared distances versus sum of absolute distances, etc.)

    method : Can be either "means","medoids", or "medians".

    Returns
    --------
    Vector of length n containing the final cluster assignments.
  """
  
  error = 0 # value of objective function at each iteration 
  obj = [] # keeps all objective function values for each iteration
  i = 0 # keep track of iteration

  r = np.zeros((n,k)) # create empty array to store cluster assignments

  # find and store k that minimize sum of square distance for each image
  newks = np.apply_along_axis(leastsquares, 1, training_data, initial_clusters, distfn)
  # create one hot coded vector for each image to signify cluster assignment
  r[range(n), newks] = 1

  # find new means
  while True:
    error = 0 # initialize error to 0
    for smallk in range(k): # iterate through clusters
      ones = np.where(r[:,smallk]==1)[0]
      if method == "means":
        means[smallk,:] = np.mean(training_data[list(ones),:], axis=0)
      elif method == "medoids":
        means[smallk,:] = training_data[list(ones),:][np.argmin(np.sum((training_data[list(ones),:] 
          - np.mean(training_data[list(ones),:], axis=0))**2,axis=1))]
      elif method == "medians":
        means[smallk,:] = np.median(training_data[list(ones),:], axis=0)
      else:
        print "Not a valid method specification"
        break

    # update responsibilities by minimizing sum of squared distances
    r_new = np.zeros((n,k))

    # stores indices of k's that minimize ssd
    newks = np.apply_along_axis(leastsquares, 1, training_data, means, distfn)
    r_new[range(n), newks] = 1

    # if none of the responsibilities change, then we've reached the optimal cluster assignments
    if np.all((r_new - r)==0):
      return r, obj
    else:
      r = r_new
    print i, r.sum(axis=0)
    i += 1
    obj.append(error)

  print 'finished'

final_responsibilities,obj = kmeans(train_images_flat, Initialize.kmeans_plusplus(k, train_images_flat, abs_sum), distfn = sumsq, method = "medoids")
print final_responsibilities.sum(axis=0)
print final_responsibilities[0]
print obj

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