#####################################################################
#                           Initialize.py                           #
#####################################################################
# Allows initialization of cluster centers                          #
# 1) using K-means++ (kmeans_plusplus)                              #
# 2) randomly (random_centers)                                      #
#                                                                   #
# Martin Reindl, Olivia Angiuli, Ty Roccca, Wilder Wohns            # 
#####################################################################

# import packages
import numpy as np
import random

"""
  Helper function that returns the closest cluster center for a given 
  data point.

  Inputs
  -------
  picture : a picture represented as a vector 784 pixels

  dist_fn : a function that calculates the distance between two pictures
    which is returned as an integer

  smallk : the number of clusters that have already been set (rest is zeros)

  Outputs
  --------
  An integer signifying the distance to the closest cluster center
"""
def _find_Dx (image, dist_fn, clusters, smallk): 
  Dx = None # initialize Dx

  # go through all clusters that have already been defined
  for i in range(smallk):
    # find distance to this cluster center
    Dx_current = dist_fn(clusters[i], image)
    # check if this cluster center is the closest cluster center
    if Dx == None or Dx_current < Dx:
      Dx = Dx_current

  return Dx



"""
  Helper function that creates a weighted probability distribution

  Inputs
  -------
  v : a vector of elements to which we want to assign weights. In our 
  case this is the vector which contains the distances of each image
  from the closest cluster center.

  Outputs
  -------
  A vector corresponding to the weights of the elements of v. In our 
  case this is a vector of length 60,000 that contains the probability 
  that a given element will be picked as a new cluster center
"""
def _weight_fn (v): 
  v = v ** 2 # square distances
  v_sum = np.sum(v) # find sum
  v = v / v_sum # find probability



"""
  Initializes cluster centers using the K-means++ algorithm as discussed in 
  http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf

  Inputs 
  -------
  k: the number clusters to be initialized

  images: the images we are assigning to the clusters in an array of shape
  60,000 x 784 (60,000 pictures with 784 pixels each)

  dist_fn: a function that calculates the distance between two images

  Outputs 
  --------
  A numpy vector of length k containing the data points which have
  been selected as cluster centers
"""
def kmeans_plusplus(k,images, dist_fn):
  clusters = np.zeros((k,784)) # matrix for initital cluster centers
  n = len(images)

  # Take one center cluster center, chosen uniformly at random from X
  clusters[0] = images[random.randint(0,n-1)]

  # define all other cluster centers
  for smallk in range(1,k):
    # For each data point x, compute D(x), the distance between x and 
    # the nearest center that has already been chosen.
    Dx = np.apply_along_axis(_find_Dx, 1, images, dist_fn, clusters, smallk)

    # Choose a new center using the Dx to generate a weighted probability
    # distibution D2
    D2 = _weight_fn(Dx) 
    clusters[smallk] = images[np.random.choice(range(n), 1, D2)[0]]

  return clusters



"""
  Initializes cluster centers randomly 

  Inputs 
  -------
  k: the number of clusters to be initialized

  Outputs 
  --------
  A numpy vector containing k randomly generated cluster centers, 
  where each cluster center is a vector of 784 pixel values 
  between 0 and 255. Per definition by the MNIST database 
  (http://yann.lecun.com/exdb/mnist/), each image is an array of 
  28 x 28 pixels, although only the inner 20 x 20 array actually 
  contains values (four pixel frame). We create such images and then
  flatten them before returning. 
"""
def random_centers(k):
  clusters = np.zeros((k,28,28)) # array to store clusters in 
  new_cluster = np.zeros((28,28)) # initiate new cluster

  # generate k new clusters and save them in clusters
  for c in range(k): 
    new_cluster[4:24,4:24] = np.floor(np.random.random_sample((20,20))*256)
    clusters[c] = new_cluster

  # flatten clusters
  clusters_flat = np.array([np.ravel(cluster) for cluster in clusters])

  return clusters_flat
