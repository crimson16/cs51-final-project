#####################################################################
#                             Load.py                               #
#####################################################################
# Reads in training and testing data and labels.				    #
# Provides training and testing data either in a non-flat (28 x 28) #
# or flat vector (1 x 784).                                         #
#                                                                   #
# Also provides a function to save image centers and random images  #
# from each cluster to file.                                        #
#                                                                   #
# Olivia Angiuli, Martin Reindl, Ty Rocca, Wilder Wohns             # 
#####################################################################

# Load packages
import os, struct
from array import array as pyarray
from cvxopt.base import matrix
import numpy as np
from numpy import append, array, int8, uint8, zeros
import Image
import random

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
    # flatten labels from format [[3],[2],[5],...] to [3,2,5,...]
    labels = [label[0] for label in labels]
    return images, labels


def save_images(k, train_images, final_responsibilities, final_clusters):
    """
    Save cluster centers, as well as randomly selected images from each
    cluster, to file.

    Inputs
    -------
    k : The number of clusters in the data.

    train_images : Data off which to train the model. In the case of
    MNIST digits, it is a vector of pixel values in format 60,000 x 784.

    final_responsibilities: A n x k vector containing one-hot-coded 
    cluster assignments for each datapoint.

    final_clusters : A k x 784 vector containing the pixel values
    for the k centers to which the clustering converged.

    Returns
    --------
    Nothing. Saves images of cluster centers and random images in each cluster
    to file.
    """
    # Processes and saves mean images and randomly selected images from each
    # cluster
    imgnum = 4 # how many images from each cluster to save
    data = np.zeros((28, 28, 3), dtype=np.uint8) # initialize uint to store data in
    cluster_data = np.zeros((28, 28, 3), dtype=np.uint8) 
    for j in range(k):  
      # Saves the mean of each cluster as an image named 'meanj.png'
      data[:,:,0] = np.reshape(final_clusters[j, :], (28,28))
      data[:,:,1] = np.reshape(final_clusters[j, :], (28,28))
      data[:,:,2] = np.reshape(final_clusters[j, :], (28,28))
      # Make an image from the uint format
      img = Image.fromarray(data, 'RGB')
      # If a directory to store the image doesn't already exist, create it
      if not os.path.exists('./cluster' + str(k)):
        os.mkdir('./cluster' + str(k))
      # Save image of cluster center as a png  
      img.save('./cluster' + str(k) + '/mean' + str(j) + '.png')

      # Randomly saves imgnum images from each cluster
      indices = np.nonzero(final_responsibilities[:, j])
      sample = random.sample(indices[0], imgnum)
      counter = 0
      
      # For each cluster, randomly create imgnum images
      for sampled_n in sample:
        cluster_data[:,:,0] = np.reshape(train_images[sampled_n, :], (28,28))
        cluster_data[:,:,1] = np.reshape(train_images[sampled_n, :], (28,28))
        cluster_data[:,:,2] = np.reshape(train_images[sampled_n, :], (28,28))
        img = Image.fromarray(cluster_data, 'RGB')
        img.save('./cluster' + str(k) + '/cluster' + str(j) + '_' + str(counter) +
          '.png')
        counter += 1