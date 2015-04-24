#####################################################################
#                             Load.py                               #
#####################################################################
# Reads in training and testing data and labels.				    #
# Provides training and testing data either in a non-flat (28 x 28) #
# or flat vector (1 x 784).  									    #                                                   #
#                                                                   #
# Olivia Angiuli, Martin Reindl, Ty Rocca, Wilder Wohns             # 
#####################################################################

# Load packages
import os, struct
from array import array as pyarray
from cvxopt.base import matrix
import numpy as np
from numpy import append, array, int8, uint8, zeros

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    # Depending on whether the training or testing dataset is needed,
    # read in the appropriate images and labels.
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    # If improper argument is provided to the "dataset" parameter,
    # raise an error.
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    # Read in labels file.
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()
    # Read in pixel values file.
    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    # Find indices of images whose labels are in the specified digit labels.
    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
    # Generate images and labels.
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels