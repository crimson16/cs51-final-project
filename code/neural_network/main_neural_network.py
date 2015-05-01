##################################
# Main Neural Ntwork Python file #
##################################

from Neural_Network import NeuralNetwork
import mnist_loader
import timeit

#Execute the neural network for comparison 
def main():
    # Time to processs dataset
    start_process = timeit.default_timer()

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    net = NeuralNetwork([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    time_neural = timeit.default_timer() - start_process

    print "The total time to run the neural network was: %d seconds" %(int(time_neural))

#Run neural network
main()