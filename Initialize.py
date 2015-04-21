"""
  Helper function that returns the closest cluster center for a given 
  data point.

  Inputs
  -------
  picture : a picture represented as an array with shape 28 x 28

  dist_fn : a function that calculates the distance between two pictures
    which is returned as an integer

  Outputs
  --------
  An integer signifying the distance to the closest cluster center
"""
def _find_Dx (image, dist_fn, clusters): 
  # initialize Dx
  Dx = None 

  # go through that have already been defined
  for smallc in clusters:
    if smallc != 0:
      # find distance to this cluster center
      Dx_current = dist_fn(image - smallc)
      # check if this cluster center is the closest cluster center
      if Dx == None or Dx_current < Dx:
        Dx = Dx_current
    else:
      return Dx



"""
  Helper function that creates a weighted probability distribution

  Inputs
  -------
  v : a vector of elements to which we want to assign weights. In our 
  case this is the vector which contains the distances of each image
  from the closest cluster center.

  Outputs
  -------
  A vector corresponding to the weights of the elements of v. In our 
  case this is the vector containing the probability that a given 
  element will be picked as a new cluster center
"""
def _weight_fn (v): 
  v = v ** 2 # square distances
  v_sum = np.sum(v) # find sum
  v = v / v_sum # find probability



"""
  Initializes cluster centers using the K-means++ algorithm as discussed in 
  http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf

  Inputs 
  -------
  k: the number clusters to be initialized

  images: the images we are assigning to the clusters

  dist_fn: a function that calculates the distance between two images

  Outputs 
  --------
  A numpy vector of length k containing the data points which have
  been selected as cluster centers
"""
def kmeans_plusplus(k,images, dist_fn):
  # matrix for initital cluster centers
  clusters = np.zeros(k)

  # Take one center cluster center, chosen uniformly at random from X
  clusters[0] = images[random.randint(0,len(images)-1)]

  # define all other cluster centers
  for smallk in range(1,k):
    # For each data point x, compute D(x), the distance between x and 
    # the nearest center that has already been chosen.
    v_find_Dx = np.vectorize(_find_Dx)
    Dx = v_find_Dx(images, dist_fn, clusters)

    # Choose a new center using the Dx to generate a weighted probability
    # distibution D2
    D2 = _weight_fn(Dx) 
    clusters[smallk] = np.random.choice(images, 1, D2)[0]

  print clusters
  return clusters



"""
  Initializes cluster centers randomly 

  Inputs 
  -------
  k: the number of clusters to be initialized

  Outputs 
  --------
  A numpy vector containing k randomly generated cluster centers, 
  where each cluster center is a 28 x 28 array of pixel values 
  between 0 and 255. Per definition by the MNIST database 
  (http://yann.lecun.com/exdb/mnist/), there is a four pixel frame. 
  Only the inner 20 x 20 array gets randomly initiated values
"""
def random_centers(k):
  clusters = np.zeros(k) # array to store clusters in 
  new_cluster = np.zeros((28,28)) # initiate new cluster

  # generate k new clusters and save them in clusters
  for c in k: 
    new_cluster[4:24,4:24] = np.floor(np.random.random_sample((24,24))*256)
    clusters[c] = new_cluster

  return clusters