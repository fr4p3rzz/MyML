import math

class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.last_input = None
        self.last_output = None
        self.delta = None

    def forward(self, inputList):
        self.last_input = inputList
        z = sum(i * w for i, w in zip(inputList, self.weights)) + self.bias
        self.last_output = self.relu(z)
        return self.last_output
    
    def relu(self, x):
        return max(0, x)

    def relu_derivative(self):
        return 1 if self.last_output > 0 else 0
    
    def store_delta(self, delta):
        self.delta = delta

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self):
        return self.last_output * (1 - self.last_output)
    
    def delta_backward(self, target):
        return (self.last_output - target) * self.relu_derivative()
        
    def update_weights(self, learning_rate=0.01):
        newWeights = []
        for w, i in zip(self.weights, self.last_input):
            newWeights.append(w - learning_rate * self.delta * i)
        self.weights = newWeights
        self.bias -= learning_rate * self.delta

    def update_weights_from_delta(self, learning_rate=0.01):
        newWeights = []
        for w, i in zip(self.weights, self.last_input):
            newWeights.append(w - learning_rate * self.delta * i)
        self.weights = newWeights