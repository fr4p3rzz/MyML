from neuron import Neuron
import random

class Layer:
    def __init__(self, inputList, activeNeurons, learning_rate=0.01):
        self.inputList = inputList
        self.learning_rate = learning_rate
        self.neurons = [
            Neuron(
                setupWeights(1, len(inputList), -0.03, 0.03)[0], 
                setupBiases(1)[0]
            )
            for _ in range(activeNeurons)
        ]
        self.output = [n.forward(inputList) for n in self.neurons]

    def forward(self, inputList):
        self.inputList = inputList
        self.output = [n.forward(inputList) for n in self.neurons]
        return self.output

    def backward(self, targetIndex):
        target_vector = [1 if i == targetIndex else 0 for i in range(len(self.neurons))]
        for neuron, target in zip(self.neurons, target_vector):
            neuron.delta = neuron.delta_backward(target)

    def backward_from_next_layer(self, next_layer):
        for i, neuron in enumerate(self.neurons):
            downstream_gradient = 0.0
            for next_neuron in next_layer.neurons:
                if i < len(next_neuron.weights):
                    downstream_gradient += next_neuron.weights[i] * next_neuron.delta
            neuron.delta = downstream_gradient * neuron.relu_derivative()

    def accumulate_gradients(self):
        for neuron in self.neurons:
            neuron.accumulate_gradients()

    def update_weights(self, batch_size):
        for neuron in self.neurons:
            neuron.update_weights(self.learning_rate, batch_size)

def setupWeights(neuronsToInitialize, requestedWeights, minValue, maxValue):
    weightList = []
    for n in range(neuronsToInitialize):
        weights = [random.uniform(minValue, maxValue) for i in range(requestedWeights)]
        weightList.append(weights)
    return weightList

def setupBiases(activeNeurons):
    return [random.uniform(0.5, 1.5) for _ in range(activeNeurons)]
