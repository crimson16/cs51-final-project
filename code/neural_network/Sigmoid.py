
################################################################################
#                             Sigmoid Functions                                #
################################################################################


import numpy as np

# The Sigmoid Functions tells us how to take a NeuralNetwork and vectorize it

def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

# Applies it to a vector
sigmoid_vec = np.vectorize(sigmoid)

# Derivative of the sigmoid function.
def sigmoid_prime(s):
    return sigmoid(s)*(1-sigmoid(s))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)