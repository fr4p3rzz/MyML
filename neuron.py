import math

class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.last_input = None
        self.last_output = None

    def forward(self, inputList):
        self.last_input = inputList
        z = sum(i * w for i, w in zip(inputList, self.weights)) + self.bias
        self.last_output = self.sigmoid(z)
        return self.last_output

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self):
        return self.last_output * (1 - self.last_output)
    
    def delta_backward(self, target):
        return (self.last_output - target) * self.sigmoid_derivative()
        
    def update_weights(self, target):
        newWeights = []
        learning_rate = 0.01
        delta = self.delta_backward(target)

        for w, i in zip(self.weights, self.last_input) :
            newWeights.append(w - learning_rate * delta * i)

        self.weights = newWeights
