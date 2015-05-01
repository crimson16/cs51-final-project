###############################################################################
#                              ClassifyClusters                               #
###############################################################################
# Allows calculating the distance between two images                          #                                
#                                                                             #
# Martin Reindl, Olivia Angiuli, Ty Roccca, Wilder Wohns                      # 
###############################################################################

import Distance
import numpy as np

"""
    classify takes clusters from our training run, and applies them to our 
    testing data. It then runs a number of accuracy functions to determine 
    how well our algorithm identifies digits. 

    Inputs
    ------
    cluster_set : clusters from our training run

    test_set : the testing data 

    test_labels : the correct digit assignments for the pictures in the 
    testing data set. 

    distfn : a function that calculates 'distance' between two images

    Outputs
    -------
    The number of clusters, the original clusters, and the accuracy of 
    our testing as a percentage.
"""

def classify(cluster_set,test_set,test_labels,distfn=Distance.sumsq):
    k = len(cluster_set)
    # Clusters is the array of final cluster means
    clusters = []
    # 
    c_index = []
    for cluster in cluster_set:
        clusters.append(cluster[0])
        c_index.append(cluster[1])
    # find number of datapoints
    n = len(test_labels) - 1

    test_clusters_asgn = np.apply_along_axis(Distance.leastsquares, 1, 
        test_set[0:n], clusters, distfn)
    test_clusters = np.array([c_index[i] for i in test_clusters_asgn])

    # correct numbers for dataset
    correct_numbers = test_labels[0:n]
    # Diffs on assignments
    diffs = test_clusters - np.array(correct_numbers)
    # Count for incorrect assignments 
    n_wrong = np.count_nonzero(diffs)

    # find out how accurate we were
    prediction_level = ((n + 1- n_wrong) / float(n + 1))*100
    print "Our accuracy in predicting test data for k = {0} was {1} %"\
        .format(k,prediction_level)
    
    # Return the cluster center
    return k, prediction_level, cluster_set
