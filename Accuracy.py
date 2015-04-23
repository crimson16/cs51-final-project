###############################################################################
#                                   Error.py                                  #
###############################################################################
# Allows initialization of cluster centers                                    #
# 1) using K-means++ (kmeans_plusplus)                                        #
# 2) randomly (random_centers)                                                #
#                                                                             #
# Martin Reindl, Olivia Angiuli, Ty Roccca, Wilder Wohns                      # 
###############################################################################

# import packages
import numpy as np
from scipy import stats
import Distance
import math 


"""
  Takes assignments from the clustering algorithm and creates a list of clusters

  Inputs
  -------
  assignments : the final cluster assignments from the clustering algorithm as 
  an array of one-hot coded vectors. For our training dataset this should be an
  array of shape 60,000 x k. 

  Outputs
  -------
  A list of length k (number of clusters) where each element is an array that 
  contains the images assigned to the cluster as indices in the original input 
  dataset.   For our training dataset this is an array of k arrays, where each 
  element of a sub-array is an integer between 0 and 60,000 - 1. 

"""
def _make_clusters(assignments):
  clusters = [] # declare list to store clusters in 

  # find clusters
  for i in range(len(assignments[0])):
    clusters.append(np.nonzero(assignments[:,i]))

  return clusters



""" 
  Finds the digit that is predominant in a cluster as well as that digit's 
  purity in the cluster. 

  Inputs 
  -------
  cluster:   An array of length k, where each element is an array that contains 
  the images assigned to the cluster as indices in the original input dataset 
  For our training dataset this is an array of k arrays, where each element 
  of a sub-array is an integer between 0 and 60,000 - 1. 

  labels : the correct digit assignments for all images. For our training 
  dataset this is a vector of length 60,000. 

  Outputs
  --------
  The digit assigned to that cluster. 

  The purity of the cluster as the percentage of the cluster that is the 
  assigned digit. 
"""
def _digit_and_purity(cluster, labels):
  e = len(cluster) # number of elements in cluster
  cluster_labels = np.zeros(k) # array to store labels for our cluster in 

  # find the labels assigned to each image in the cluster
  for i in range(e):
    cluster_labels[i] = labels[cluster[e]]

  # find most common digit and its frequency 
  digit, digit_count = stats.mode(cluster)
  digit = digit[0] 
  digit_count = digit_count[0] 
  purity = digit_count/k 

  return digit, purity



"""
  Finds the standard deviation for a given cluster (can't use np.std because 
  cluster center does not have to be mean)

  Inputs 
  -------
  cluster: An array of length k, where each element is an array that contains 
  the images assigned to the cluster as indices in the original input dataset 
  For our training dataset this is an array of k arrays, where each element 
  of a sub-array is an integer between 0 and 60,000 - 1. 

  center : the cluster center from our clustering algorithm

  Outputs 
  -------
  The standard deviation of the cluster
"""
def _std(cluster, center, images):
  e = len(cluster) # number of elements in cluster
  cluster_images = np.zeros(k) # array to store the actual cluster images in 

  # retrieve cluster images
  for i in range(e):
    cluster_images[i] = images[cluster[i]]

  # find squared distances from center for each cluster
  find_distance = np.vectorize(Distance.sumsq)
  distances = find_distance(cluster_images,center)

  #find standard deviation
  return math.sqrt(np.sum(distances)/e)
  

"""
  Prints the purity and standard deviation of each digit 

  Inputs 
  -------
  assignments : the final cluster assignments from the clustering algorithm as 
  an array of one-hot coded vectors. For our training dataset this should be an
  array of shape 60,000 x k. 

  labels : the correct digit assignments for all images. For our training 
  dataset this is a vector of length 60,000. 

  cluster_centers : the cluster centers from our clustering algorithm

  Outputs 
  -------
  Prints results as "Digit: x, Purity: y%, Standard Deviation:z, Count: a"

  Returns results as an array of size 10 x 4, where each row holds information
  in the form [digit, purity, std, count] for the corresponding digit
"""
def final_accuracy(assignments, labels, images, cluster_centers): 
  clusters = _make_clusters(assignments) # make clusters
  k = len(clusters) # number of clusters

  # info holds information about each cluster as digit, purity, std, size
  info = np.zeros((k,4)) 

  # loop through clusters
  for i in range(k):
    # find digit and purity for this cluster
    info[i][:2] = _digit_and_purity(clusters[i], labels)
    # find standard deviation for this cluster
    info[i][2] = _std(clusters[i], cluster_centers[i], images)
    #find number of elements in cluster
    info[i][3] = len(clusters[i])

  # initialize print_array to hold final information 
  print_array = np.zeros((10,4))
  # if more clusters than digits, we have to create weighted averages 
  # for the final results
  if k > 10: 
    for i in range(10):
      # filter clusters to only contain clusters with digit i 
      condition = clusters[:,0] == i
      digit_clusters = np.compress(condition,clusters,axis=0)

      # create weighted averages and save them in print_array
      weights = info[:,3]/np.sum(info[:,3])

      print_array[i][0] = i # set digit
      print_array[i][1] = np.sum(weights * info[:,1]) # find weighted purity
      print_array[i][2] = np.sum(weights * info[:,2]) # find weighted std
      print_array[i][3] = np.sum(info[:,3]) # find total count

      # print to command line 
      print "Digit: %d  Purity: %f  Standard Deviation: %f  Count: %d" % (
        print_array[i][0], print_array[i][1],print_array[i][2],print_array[i][3])
  else:
    raise ValueError("Minimum cluster number is 10")

  return print_array



















