import numpy as np
import math


class NeuralNetwork()：
    def __init__(self):
        np.random.seed(1)

        self.synapticWeights = 2 * np.random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1.0 / (1 + math.exp(-x))

    def predict(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synapticWeights))

    def train(self, trainingSet_inputs, trainingSet_outputs, numberOfIterations):


if __name__ == '__main__'：
# initialize a NN
    neuralNetwork = NeuralNetwork()

    print 'Random starting synaptic weights:'
    print neuralNetwork.synapticWeights

    # Training Set
    trainingSet_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    trainingSet_outputs = np.array([0, 1, 1, 0]).T

    # train the network for 10000 times
    neuralNetwork.train(trainingSet_inputs, trainingSet_outputs, 10000)

    print 'Trained weights afterwards'
    print neuralNetwork.synapticWeights

    print 'predicting for [1, 0, 0] -> ?: '
    print neuralNetwork.predict(np.array([1, 0, 0]))
