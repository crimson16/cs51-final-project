###############################################################################
#                                  Distance.py                                #
###############################################################################
# Allows calculating the distance between two images                          #                                #
#                                                                             #
# Martin Reindl, Olivia Angiuli, Ty Roccca, Wilder Wohns                      # 
###############################################################################


# iterates over means matrix to calculate sum of differences
def diff_sum(xn, m):
    diff = xn - m
    return diff.sum(axis=0)

# iterates over means matrix to calculate sum of absolute differences
def abs_sum(xn, m):
    diff = abs(xn - m)
    return diff.sum(axis=0)

# iterates over means matrix to calculate sum of squared distances
def sumsq(xn, m):
    diff = (xn - m) ** 2
    return diff.sum(axis=0)

# iterates over means matrix to calculate maximum squared distance
def maxdist(xn, m):
    diff = (xn - m) ** 2
    return diff.max(axis=0)

# returns index of cluster that minimizes error
def leastsquares(xn, means, dist_fn):
    # Pass in a distance function, and then find the distance, via that measure,
    # between every point and each cluster center
    errors = np.apply_along_axis(dist_fn, 1, means, xn) 
    # For each datapoint, find the minimum distance cluster center.
    return np.argmin(errors)