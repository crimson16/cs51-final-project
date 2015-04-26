
####################
# ClassifyClusters #
####################

'''

ClassifyClusters is used to process our cluster. Right now 
we have clusters, however, we do not use them for anything. What
ClassifyClusters will do is it will take the test set and based on
a naive implementation of knn we will be able to see how well we can
identify points in the dataset

Inputs
------
Clusters, Test dataset

Outputs
-------
Accuracy of our classifier, which matches the test images to
the training data 

What I am doing is essentially like taking final Accuracy and then applying
it to our data set in a knn style

'''
import Distance
import numpy as np
def classify(cluster_set, test_set, test_labels, distfn = Distance.sumsq, n = None):

    k = len(cluster_set)
    # Clusters is the array of final cluster means
    clusters = []
    # 
    c_index = []
    for cluster in cluster_set:
        clusters.append(cluster[0])
        c_index.append(cluster[1])

    if n == None:
        n = len(test_labels) - 1
    test_clusters_asgn = np.apply_along_axis(Distance.leastsquares, 1, test_set[0:n],
        clusters, distfn)
    test_clusters = np.array([c_index[i] for i in test_clusters_asgn])

    # correct numbers for dataset
    correct_numbers = test_labels[0:n]
    # Diffs on assignments
    diffs = test_clusters - np.array(correct_numbers)
    # Count for incorrect assignments 
    n_wrong = np.count_nonzero(diffs)

    prediction_level = ((n - n_wrong) / float(n))*100
    print "Our accuracy in predicting test data for k = {0} was {1} %".format(k,prediction_level)
    
    # Return the cluster center
    return k, prediction_level, cluster_set

    






# now for each test image, calculate the closest center
# def closest_center(image, clusters, cluster_assignments):
    # return Distance.leastsquares(image, clusters, Distance.sumsq)

# print closest_center(test_images_flat[0], final_clusters, cluster_assignments)

# for image in test_images
#     calculate the closest center
#     determine if the assignment of that center is the same as actual
#     report % correct