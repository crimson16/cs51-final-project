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



"""
  Takes assignments from the clustering algorithm and creates an array of 
  length k (number of clusters), 

  Inputs
  -------
  assignments : the final cluster assignments from the clustering algorithm as 
  an array of one-hot coded vectors. For our training dataset this should be an
  array of shape 60,000 x k. 

  Outputs
  -------
  An array of length k, where each element is an array that contains 
  the images assigned to the cluster as indices in the original input dataset. 
  For our training dataset this is an array of k arrays, where each element 
  of a sub-array is an integer between 0 and 60,000 - 1. 

"""
def _make_clusters(assignments):
  




""" 
  Finds the digit that is predominant in a given cluster as well as that digits 
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
def digit_and_purity(cluster, labels):
  k = len(assignments[0]) # the number of clusters

  # split up th


"""
  Finds the standard deviation for a given cluster

  Inputs 
  -------
  cluster: An array of length k, where each element is an array that contains 
  the images assigned to the cluster as indices in the original input dataset 
  For our training dataset this is an array of k arrays, where each element 
  of a sub-array is an integer between 0 and 60,000 - 1. 

  Outputs 
  -------
  The standard deviation of the cluster
"""
def std_deviation(cluster):


"""
  Prints the purity and standard deviation of each digit

  Inputs 
  -------
  assignments : the final cluster assignments from the clustering algorithm as 
  an array of one-hot coded vectors. For our training dataset this should be an
  array of shape 60,000 x k. 

  labels : the correct digit assignments for all images. For our training 
  dataset this is a vector of length 60,000. 

  Outputs 
  -------
  None (prints results as digit: x, purity: y%, standard deviation: z)
"""
def print_results(assignments, labels): 



