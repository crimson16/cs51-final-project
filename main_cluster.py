

# Helper Packages
import numpy as np
import os, struct,random #,
import sys
from numpy import append, array, int8, uint8, zeros
import base64
import json
import timeit

# Created functions
import Initialize
import Accuracy
import Distance
import Load
import Kmeans
import ClassifyClusters


#########################
# Set parameter values  #
#########################
k = int(sys.argv[1]) # number of clusters (system argument)
method = "means"
init_type = "random"
prop = 5

# Process Command Line Arguments
arg_num = len(sys.argv)

if arg_num > 5:
    raise ValueError("4 argument max")
else:
    if arg_num == 5:
        if int(sys.argv[4]) > 0 and int(sys.argv[4]) <= 100:
            prop = int(sys.argv[4])
    if arg_num >= 4:
        if sys.argv[3] == "kplusplus":
            init_type = "kplusplus"
    if arg_num >=3:
        if str(sys.argv[2]) in ["means", "medoids", "medians"]:
            method = str(sys.argv[2])

######################################
# Load in training images and labels #
######################################

# load training and testing images and labels as 60,000 x 28 x 28 array
train_images,train_labels = Load.load_mnist("training",path=os.getcwd(), prop = prop)
test_images,test_labels = Load.load_mnist("testing",path=os.getcwd())

# flatten training images into 60,000 x 784 array
train_images_flat = np.array([np.ravel(img) for img in train_images])
test_images_flat = np.array([np.ravel(img) for img in test_images])

def main (k, m="means", init_type="random"):
    
    # Starting clustering timer

    start_cluster = timeit.default_timer()
    
    # Process arguments
    if k < 10:
        raise ValueError("Minimum cluster number is 10")
    
    #Process method of clustering
    if m not in ["means", "medoids", "medians"]:
        raise ValueError("Not a valid method specification; must be 'means',\
          'medoids', or 'medians'")
    
    # Method for clustering
    initial_clusters = None
    if init_type == "random":
        initial_clusters = Initialize.random_centers(k)
    else:
        init_type = "kplusplus"
        initial_clusters = Initialize.kmeans_plusplus(k, train_images_flat,
                                           dist_fn=Distance.sumsq)
        
        
        
    # Run clustering algorithm
    final_responsibilities, final_clusters = Kmeans.kmeans(k, train_images_flat,
        initial_clusters, distfn = Distance.sumsq, method=m)
    
    # Output of results
    print final_responsibilities.sum(axis=0)

    # Time to cluster
    end_cluster = timeit.default_timer()
    clustering_time = end_cluster - start_cluster
    print "Time spent clustering : ", clustering_time


    # Save representative images to file.

    title = m + "_" + init_type + "_cluster" + str(k)
    Load.save_images(k, train_images, final_responsibilities, 
                     final_clusters, title)

    # Calculate final accuracy for clusters
    final, cluster_set = Accuracy.final_accuracy(final_responsibilities, 
        train_labels, train_images_flat, final_clusters)

    # Now see how well we can classify the dataset
    start_cluster_test = timeit.default_timer()
    predictions = ClassifyClusters.classify(cluster_set, test_images_flat, 
        test_labels, distfn = Distance.sumsq, n=None)
    finish_cluster_test = timeit.default_timer()

    testing_time = finish_cluster_test - start_cluster_test

    print "Time spent testing : ", testing_time

    ###########
    # Outputs #
    ###########

    # Serializing numpy array - from below source 
    # http://stackoverflow.com/questions/3488934/simplejson-and-numpy-array

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            """
            if input object is a ndarray it will be converted into a 
            dict holding dtype, shape and the data base64 encoded
            """
            if isinstance(obj, np.ndarray):
                data_b64 = base64.b64encode(obj.data)
                return dict(__ndarray__=data_b64,
                            dtype=str(obj.dtype),
                            shape=obj.shape)
            # Let the base class default method raise the TypeError
            return json.JSONEncoder(self, obj)


    def json_numpy_obj_hook(dct):
        """
        Decodes a previously encoded numpy ndarray
        with proper shape and dtype
        :param dct: (dict) json encoded ndarray
        :return: (ndarray) if input was an encoded ndarray
        """
        if isinstance(dct, dict) and '__ndarray__' in dct:
            data = base64.b64decode(dct['__ndarray__'])
            return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
        return dct


    # k, prediction level, cluster_set, 
    results = {"k" : k, "prediction_accuracy" : predictions[1], 
    "cluster_means" : cluster_set, "cluster_stats" : final,
    "clustering_time" : clustering_time, "testing_time" : testing_time}


    with open('./' + title + '/' + title + '_results.json', 'w') as outfile:
        json.dump(results, outfile, cls=NumpyEncoder)


####################
# Call to function #
####################
main(k, m=method, init_type=init_type)


