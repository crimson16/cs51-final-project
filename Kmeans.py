#####################################################################
#                             Kmeans.py                             #
#####################################################################
# Runs a handwritten version of the k-means algorithm.              #
# Takes in a vector of training data and of initial                 #
#                                                                   #
# Olivia Angiuli, Martin Reindl, Ty Rocca, Wilder Wohns             # 
#####################################################################

import numpy as np
import Distance


def kmeans(k, training_data, initial_clusters, distfn = Distance.sumsq,
  method = "means"):
  """
    Run the k-means (if method = "means") or k-medians
    (if method = "medians"), or k-medoids (if method = "medoids"), 
    algorithm, based off of Lloyd's algorithm.
    
    Inputs
    -------
    training_data : Data off which to train the model. In the case of
    MNIST digits, it is a vector of pixel values in format 60,000 x 784

    initial_clusters : Vector containing the initial cluster centers
    These can either be initialized randomly or by k-means++ and have
    format 60,000 x 784.

    distfn : A function tdhat measures distance between points (i.e. sum of
    squared distances versus sum of absolute distances, etc.)

    method : Can be either "means","medoids", or "medians".

    Returns
    --------
    final_responsibilities : A n x k vector containing one-hot-coded
    cluster assignments for each datapoint.
    final_cluster_centers: A k x 784 vector containing the pixel values
    for the k centers to which the clustering converged.
  """
  
  n = len(training_data) # number of training instances

  i = 0 # keep track of iteration

  r = np.zeros((n,k)) # create empty array to store cluster assignments

  # find and store k that minimize sum of square distance for each image
  newks = np.apply_along_axis(Distance.leastsquares, 1, training_data,
    initial_clusters, distfn)

  # create one hot coded vector for each image to signify cluster assignment
  r[range(n), newks] = 1

  # create a "means" vector to store cluster centers as they are updated
  means = initial_clusters

  # Find new means
  while True:
    for smallk in range(k): # iterate through clusters
      ones = np.where(r[:,smallk]==1)[0]
      print ones
      # The k-means method updates cluster centers as being the mean of each
      # corresponding pixel of the datapoints that are contained in that
      # cluster.
      if method == "means":
        means[smallk,:] = np.mean(training_data[list(ones),:], axis=0)
      # The k-medoids method updates cluster centers as being the closest
      # *datapoint* to the mean of the corresponding pixel of datapoints
      # contained in that cluster.
      elif method == "medoids":
        dist_to_ctr = np.sum((training_data[list(ones),:] 
          - np.mean(training_data[list(ones),:], axis=0))**2,axis=1)
        means[smallk,:] = training_data[list(ones),:][np.argmin(dist_to_ctr)]
      # The k-medians method updates cluster centeras as being the median of
      # each corresponding pixel of the datapoints that are contained in that
      # cluster.
      elif method == "medians":
        means[smallk,:] = np.median(training_data[list(ones),:], axis=0)
      # If no proper value is chosen for method, then return error.
      else:
        raise ValueError("Not a valid method specification; must be 'means',\
          'medoids', or 'medians'")

    # update responsibilities by minimizing distance metric
    r_new = np.zeros((n,k))

    # stores indices of k's that minimize distance metric
    newks = np.apply_along_axis(Distance.leastsquares, 1, training_data,
      means, distfn)
    r_new[range(n), newks] = 1

    # if none of the responsibilities change, then we've reached the optimal
    # cluster assignments
    if np.all((r_new - r)==0):
      return r, means
    else:
      r = r_new
    # After each iteration, print iteration number and the number of images
    # assigned to a given cluster.
    print i, r.sum(axis=0)
    i += 1

  print 'finished'