####
# kmeans_raw.py
#
# Implements k-means clustering without k-means++.
# Uses MNIST handwritten digit training dataset.
# Takes in a value of k (# of clusters) as a system argument.
#
# Olivia Angiuli, Martin Reindl, Ty Rocca, Wilder Wohns
###

#################
# Load packages #
#################

import numpy as np
import os, struct,random,sys
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

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

##########################
## Function definitions ##
##########################

# iterates over means matrix to calculate sum of squared distances
def iteratemeans(m, xn):
    diff = (xn - m) ** 2
    return diff.sum(axis=0)

# returns index of cluster that minimizes error
def leastsquares(xn, means):
    global error
    errors = np.apply_along_axis(iteratemeans, 1, means, xn) 
    error += errors[np.argmin(errors)]
    return np.argmin(errors)


######################################
# Load in training images and labels #
######################################
train_images,train_labels = load_mnist("training",path=os.getcwd())
train_images_flat = np.array([np.ravel(img) for img in train_images])


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

# Randomly initialize cluster centers (note: not k-means++)
for i in range(n):
  r[i][random.randint(0,k-1)] = 1

#################################################
# Lloyd's algorithm -- find optimal clustering  #
#################################################

# value of objective function at each iteration
error = 0
# keeps all objective function values for each iteration
obj = []

i = 0
r_new = np.zeros((n,k)) # newly assigned responsibilities: initialized to 0 and updated witin the loop
while True:
  error = 0
  # update means
  for smallk in range(k):
    ones = np.where(r[:,smallk]==1)[0]
    means[smallk,:] = np.mean(train_images_flat[list(ones),:], axis=0)
  # update responsibilities by minimizing sum of squared distances
  r_new = np.zeros((n,k))
  # stores indices of k's that minimize ssd
  newks = np.apply_along_axis(leastsquares, 1, train_images_flat, means)
  r_new[range(n), newks] = 1
  # if none of the responsibilities change, then we've reached the optimal cluster assignments
  if np.all((r_new - r)==0):
    break
  else:
    r = r_new
  print i, r.sum(axis=0)
  i += 1
  obj.append(error)

print 'finished'
total = time.time() - start
print total


# processes and saves mean images and randomly selected images from each cluster
'''
imgnum = 4
data = np.zeros((32, 32, 3), dtype=np.uint8)
cluster_data = np.zeros((32, 32, 3), dtype=np.uint8)
for j in range(K):  
  # saves the mean of each cluster as an image named 'K_meanj.png'
  data[:,:,0] = np.reshape(means[j, 0:1024], (32,32))
  data[:,:,1] = np.reshape(means[j, 1024:2048], (32,32))
  data[:,:,2] = np.reshape(means[j, 2048:3072], (32,32))
  img = Image.fromarray(data, 'RGB')
  img.save('./cluster' + str(K) + '/mean' + str(j) + '.png')

  # randomly saves imgnum images from each cluster
  indices = np.nonzero(r[:, j])
  sample = random.sample(indices[0], imgnum)
  counter = 0
  
  for n in sample:
    cluster_data[:,:,0] = np.reshape(x[n, 0:1024], (32,32))
    cluster_data[:,:,1] = np.reshape(x[n, 1024:2048], (32,32))
    cluster_data[:,:,2] = np.reshape(x[n, 2048:3072], (32,32))
    img = Image.fromarray(cluster_data, 'RGB')
    img.save('./cluster' + str(K) + '/cluster' + str(j) + '_' + str(counter) + '.png')
    counter += 1

'''









