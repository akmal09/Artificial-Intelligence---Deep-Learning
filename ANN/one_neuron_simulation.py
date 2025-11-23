from Neuron import Neuron
import numpy as np

weights = np.array([0,1])
bias = 4


neuron = Neuron(weights, bias)

dataSet = np.array([2,3])
print(neuron.feedForward(dataSet))