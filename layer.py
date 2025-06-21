from neuron import Neuron
import random

class Layer:
    def __init__(self, inputList, activeNeurons):
        self.inputList = inputList
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


def setupWeights(neuronsToInitialize, requestedWeights, minValue, maxValue):
    weightList = []
    for n in range(neuronsToInitialize):
        weights = [random.uniform(minValue, maxValue) for i in range(requestedWeights)]
        weightList.append(weights)
    return weightList

def setupBiases(activeNeurons):
    return [random.uniform(0.5, 1.5) for _ in range(activeNeurons)]