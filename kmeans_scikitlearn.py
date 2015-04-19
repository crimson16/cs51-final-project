####
# kmeans_scikitlearn.py
#
# Implements k-means clustering using scikit_learn.
# Meant to act as a comparison method to our raw implementation
# of k-means clustering.
#
# Olivia Angiuli, Martin Reindl, Ty Rocca, Wilder Wohns
###


from time import time
import os, struct,random,sys
import numpy as np
import matplotlib.pyplot as plt

from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

from sklearn.cluster import KMeans

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

######################################
# Load in training images and labels #
######################################
train_images,train_labels = load_mnist("training",path=os.getcwd())
train_images_flat = np.array([np.ravel(img) for img in train_images])

test_images,test_labels = load_mnist("testing",path=os.getcwd())
test_images_flat = np.array([np.ravel(img) for img in test_images])



#########################
# Set parameter values  #
#########################
k = int(sys.argv[1]) # number of clusters (system argument)
n = len(train_images) # number of data points
l2 = len(train_images[0][0]) # number of rows in a training datapoint (assumes each training datapoint is same size)
l1 = len(train_images[0]) # number of columns in a single training datapoint (assumes each training datapoint is same size)
l = l1 * l2 # total number of pixels in a training datapoint
r = np.zeros((n,k)) # matrix of responsibilities (assignments of each datapoint to a cluster)
means = np.zeros((k,l))

kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
kmeans.fit(train_images_flat)
#fit = sklearn.fit(train_images_flat)

# Obtain predictions for each point.
Z = kmeans.predict(test_images_flat)

# For a given cluster, take its "labelling" in order to be whatever
# true class the majority of the elements belong to.
#
# Use this method in order to assign a "label" to each of the clusters,
# as represented by the cluster_assignments list.

cluster_assignments = []
cluster_accuracies = []
for assigned_cluster in range(k):
	cluster_points = np.where(Z==assigned_cluster)[0]
	cluster_true_labels = test_labels[cluster_points]
	cluster_true_labels = [int(label[0]) for label in cluster_true_labels]
	label_counts = np.bincount(cluster_true_labels)
	cluster_assignments.append(np.argmax(label_counts))
	cluster_accuracies.append(float(np.argmax(label_counts))/sum(label_counts))

print cluster_assignments
print cluster_accuracies # is this correct?





