###############################################################################
#                                   Accuracy.py                                  #
###############################################################################
# Allows initialization of cluster centers                                    #
# 1) using K-means++ (kmeans_plusplus)                                        #
# 2) randomly (random_centers)                                                #
#                                                                             #
# Martin Reindl, Olivia Angiuli, Ty Rocca, Wilder Wohns                      # 
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
    clusters.append(np.nonzero(assignments[:,i])[0])

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
  cluster_labels = np.zeros(e) # array to store labels for our cluster in 

  # find the labels assigned to each image in the cluster
  for i in range(e):
    cluster_labels[i] = labels[cluster[i]]

  # find most common digit and its frequency 
  digit, digit_count = stats.mode(cluster_labels)
  digit = digit[0] 
  digit_count = digit_count[0] 
  purity = digit_count/e 

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
    cluster_images = np.zeros((e,784)) # array to store the actual cluster images in 
  
    # retrieve cluster images
    for i in range(e):
        cluster_images[i] = images[cluster[i]]

    # find squared distances from center for each cluster
    distances = np.apply_along_axis(Distance.sumsq, 1, cluster_images, center)

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

    # cluster_set holds information about each cluster as cluster, means, digit
    cluster_set = [[()] for i in xrange(k)]

    # loop through clusters
    for i in range(k):
        # find digit and purity for this cluster
        info[i][:2] = _digit_and_purity(clusters[i], labels)
        # find standard deviation for this cluster
        info[i][2] = _std(clusters[i], cluster_centers[i], images)
        #find number of elements in cluster
        info[i][3] = len(clusters[i])

        # Make the cluster dataset, to use for classifying 
        cluster_set[i] = cluster_centers[i], info[i][0]


    # initialize print_array to hold final information 
    final = np.zeros((10,4))
    # if more clusters than digits, we have to create weighted averages 
    # for the final results
    if k >= 10: 
        for i in range(10):
            # filter info to only contain clusters with digit i 
            condition = info[:,0] == i
            digit_info = np.compress(condition,info,axis=0)

            # create weighted averages and save them in final
            weights = digit_info[:,3]/np.sum(digit_info[:,3])

            final[i][0] = i # set digit
            final[i][1] = np.sum(weights * digit_info[:,1]) # find weighted purity
            final[i][2] = np.sum(weights * digit_info[:,2]) # find weighted std
            final[i][3] = np.sum(digit_info[:,3]) # find total count

            # print to command line 
            print "Digit: %d  Purity: %4.1f" % (final[i][0], final[i][1]*100) + "%" + \
            " SD: %6.1f  Count: %d" % (final[i][2], final[i][3])
    else:
        raise ValueError("Minimum cluster number is 10")

    return final, cluster_set


