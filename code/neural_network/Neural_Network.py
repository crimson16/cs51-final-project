"""

This neural network is to be used to compare to our k-means model. We hope that 
this model is a good comparison. The code is made by following the implemention by 
http://neuralnetworksanddeeplearning.com/chap1.html

"""


# Useful libraries
import numpy as np
import random
# Our Defined Libraries
import Sigmoid


################################################################################
#                             Neural Network Class                             #
################################################################################



class NeuralNetwork():

    # In the initialization of the Neural Network we need a list describing the 
    # neural network, each item represents number of neurons in that level
    def __init__ (self, n_settings):

        # Neural network settings
        self.n_settings = n_settings
        self.layer_count = len(n_settings)

        # Weights and Biases are assigned randomly, can improve later!
        self.biases = [np.random.randn(b, 1) for b in n_settings[1:]]
        self.weights = [np.random.randn(y, x) for x, y in 
                        zip(n_settings[:-1], n_settings[1:])]

    def processinputs(self, iput):
        ''' returns outputs in iput is an input '''
        for bias, weight in zip(self.biases, self.weights):
            iput = Sigmoid.sigmoid_vec(np.dot(weight, iput) + bias)
        return iput

    # The way that learning occurs is that we have to use a stochastic gradient
    # descent
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)

        
        for j in xrange(epochs): 
            # For each of the epoch we shuffle data
            random.shuffle(training_data)
            # We then take the shuffled data and split the 
            # data into subsets of data
            mini_batches = [training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            # now apply the a setp of gradient descent
            for mini_batch in mini_batches:
                #update baches
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, 
                    self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # process inputs
        activation = x
        # list to store all the activations, layer by layer
        activations = [x] 
        # list to store all the z vectors, layer by layer
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Sigmoid.sigmoid_vec(z)
            activations.append(activation)
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            Sigmoid.sigmoid_prime_vec(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Here we iterate from the lowest neural layer (layer = 1 is lowest) 
        # all the way up to the highest neural layer 
        for layer in xrange(2, self.layer_count):
            z = zs[-layer]
            spv = Sigmoid.sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * spv
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. The neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.processinputs(x)), y) 
                            for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)





