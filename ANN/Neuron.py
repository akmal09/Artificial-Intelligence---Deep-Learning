import numpy as np

def sigmoid(zTotal):
    # activations function = 1 / (1+e^(zTotal))
    return 1 / (1+np.exp(-zTotal))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

#   For processing data input
    def feedForward(self, inputs):
        zTotal = np.dot(self.weights, inputs) + self.bias
        activation = sigmoid(zTotal)
        return activation



