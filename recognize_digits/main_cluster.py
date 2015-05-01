
###############################################################################
#                                   Main.py                                   #
###############################################################################
# Main.py lies at the center of our clustering. It combines several other     #
# modules (see below) to check user input and then follow the four step       #
# implementation of clustering we outlined in our proposal:                   #
# 1) Load Data                                                                #
# 2) Initialize Clusters                                                      #
# 3) Run Classification Algorithm                                             #
# 4) Find Accuracy                                                            #
###############################################################################

###############################################################################
#                               Import Packages                               #
###############################################################################
# import numpy and abbreviate because of common use
import numpy as np
# import other helper modules
import os, struct, random, sys, base64, json, timeit
# Import our modules
import Initialize, Accuracy, Distance, File, Kmeans, ClassifyClusters


###############################################################################
#                Set parameter values and check user input                    #
###############################################################################

# set default command line arguments 
k = 20
init_vis = "random"
method = "means"
prop = 100

# make sure user submitted a correct number of arguments
arg_num = len(sys.argv) 
if arg_num > 5:
    raise ValueError("Too many arguments supplied")
# check user input and update defaults
else:
    # set proportion
    if arg_num == 5:
        if int(sys.argv[4]) > 0 and int(sys.argv[4]) <= 100:
            prop = int(sys.argv[4])
        else: 
            raise ValueError("Please select a proportion between 0 and 100")
    # set initialization
    if arg_num >= 4:
        if sys.argv[3] == "kplusplus":
            init_type = "kplusplus"
        elif sys.argv[3] == "random": 
            init_type = "random"
        else: 
            raise ValueError("Method must be \'random\' or \'kplusplus\'")
    # set method 
    if arg_num >= 3:
        if str(sys.argv[2]) in ["means", "medoids", "medians"]:
            method = str(sys.argv[2])
        else: 
            raise ValueError("Method must be \'means\', \'medians\' or \
            \'medoids\'")
    # set number of clusters
    if arg_num >= 2: 
        print sys.argv[1]
        if int(sys.argv[1]) < 10 or int(sys.argv[1]) >= 50: 
            raise ValueError("Number of clusters must be between 10 and 50")
        else: 
            k = int(sys.argv[1])


# show user what s/he is running
print "Running %s algorithm with %s initialization, %d percent of the dataset\
, and %d clusters" % (method, init_type, prop, k)

###############################################################################
#                                 Load Data                                   #
###############################################################################

# load training and testing images and labels as 60,000 x 28 x 28 array
train_images,train_labels=File.load_mnist("training",path=os.getcwd(),prop=prop)
test_images,test_labels = File.load_mnist("testing",path=os.getcwd())

# flatten training images into 60,000 x 784 array
train_images_flat = np.array([np.ravel(img) for img in train_images])
test_images_flat = np.array([np.ravel(img) for img in test_images])


###############################################################################
#                       Run Classification Algorithm                          #
###############################################################################
"""
    Runs the classification algorithm with conditions supplied

    Inputs 
    -------
    k : the number of clusters

    m : the type of clustering (either 'means', 'medians', or 'medoids')

    init_type : type of initialization (either 'random' or 'kmeans_plusplus')

    Outputs
    -------
    Prints time values, as well as major steps of the algorithm for user to 
    follow along. 
"""
def main (k, m="means", init_type="random"):
    # Starting clustering timer
    start_cluster = timeit.default_timer()

    # Initialize clusters
    if init_type == "random":
        initial_clusters = Initialize.random_centers(k)
    else:
        init_type = "kplusplus"
        initial_clusters = Initialize.kmeans_plusplus(k, train_images_flat,\
            dist_fn=Distance.sumsq)
        
    # Run clustering algorithm
    final_responsibilities, final_clusters = Kmeans.kmeans(k,train_images_flat,
        initial_clusters, distfn = Distance.sumsq, method=m)

    # Find and print clustering time
    end_cluster = timeit.default_timer()
    clustering_time = end_cluster - start_cluster
    print "Time spent clustering : ", clustering_time

    # Save representative images to file.
    title = m + "_" + init_type + "_cluster" + str(k)
    File.save_images(k, train_images, final_responsibilities, 
                     final_clusters, title)

    ###########################################################################
    #                           Calculate Accuracy                            #
    ###########################################################################

    # Calculate final accuracy for clusters
    final, cluster_set = Accuracy.final_accuracy(final_responsibilities, 
        train_labels, train_images_flat, final_clusters)

    # Now see how well we can classify the dataset
    start_cluster_test = timeit.default_timer()
    predictions = ClassifyClusters.classify(cluster_set, test_images_flat, 
        test_labels, distfn = Distance.sumsq)
    finish_cluster_test = timeit.default_timer()

    # find time it took to test 
    testing_time = finish_cluster_test - start_cluster_test
    print "Time spent testing : ", testing_time

    ###########################################################################
    #                                 Outputs                                 #
    ###########################################################################

    # k, prediction level, cluster_set, 
    results = {"k" : k, "prediction_accuracy" : predictions[1], 
    "cluster_means" : cluster_set, "cluster_stats" : final,
    "clustering_time" : clustering_time, "testing_time" : testing_time}

    with open('./results/' + title + '/' + title + '_results.json', 'w') as outfile:
        json.dump(results, outfile, cls=File.NumpyEncoder)

###############################################################################
#                               Call to Function                              #
###############################################################################
main(k, m=method, init_type=init_type)


