###############################################################################
#                             Load.py                                         #
###############################################################################
# Reads in training and testing data and labels.				              #
# Provides training and testing data either in a non-flat (28 x 28)           #
# or flat vector (1 x 784).                                                   #
#                                                                             #
# Also provides a function to save image centers and random images            #
# from each cluster to file.                                                  #
#                                                                             #
# Olivia Angiuli, Martin Reindl, Ty Rocca, Wilder Wohns                       # 
###############################################################################

# Load packages
import os, struct
from array import array as pyarray
from cvxopt.base import matrix
import numpy as np
from numpy import append, array, int8, uint8, zeros
from PIL import Image
import random, json, base64

def load_mnist(dataset="training", digits=np.arange(10), path=".", prop = 100):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    # Depending on whether the training or testing dataset is needed,
    # read in the appropriate images and labels.
    if dataset == "training":
        fname_img = os.path.join(path, '../data/train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, '../data/train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, '../data/t10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, '../data/t10k-labels-idx1-ubyte')
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
    N = int(len(ind) * prop/100.)
    # Generate images and labels.
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(int(len(ind) * prop/100.)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    # flatten labels from format [[3],[2],[5],...] to [3,2,5,...]
    labels = [label[0] for label in labels]
    return images, labels


def save_images(k, train_images, final_responsibilities, final_clusters,title):
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

    title : The title for the output for the cluster
        format is Method_Init-type_Cluster_K

    Returns
    --------
    Nothing. Saves images of cluster centers and random images in each cluster
    to file.
    """
    # Processes and saves mean images and randomly selected images from each
    # cluster
    imgnum = 4 # how many images from each cluster to save
    data = np.zeros((28, 28, 3), dtype=np.uint8)# initialize uint to store data
    cluster_data = np.zeros((28, 28, 3), dtype=np.uint8) 
    for j in range(k):  
        # Saves the mean of each cluster as an image named 'meanj.png'
        data[:,:,0] = np.reshape(final_clusters[j, :], (28,28))
        data[:,:,1] = np.reshape(final_clusters[j, :], (28,28))
        data[:,:,2] = np.reshape(final_clusters[j, :], (28,28))
        # Make an image from the uint format
        img = Image.fromarray(data, 'RGB')
        # If a directory to store the image doesn't already exist, create it
        if not os.path.exists('./results'):
            os.mkdir('./results')
        if not os.path.exists('./results/' + title):
            os.mkdir('./results/' + title)
        # Save image of cluster center as a png  
        img.save('./results/' + title + '/mean' + str(j) + '.png')

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
            img.save('./results/' + title + '/cluster' + str(j) + '_' + str(counter) +
                '.png')
            counter += 1



# Serializing numpy array - from below source 
# http://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
"""
    If input object is a ndarray it will be converted into a 
    dict holding dtype, shape and the data base64 encoded
"""
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(obj.data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)

"""
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
"""
def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct