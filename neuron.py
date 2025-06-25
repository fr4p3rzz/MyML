import math

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.last_input = None
        self.last_output = None
        self.delta = None

        self.grad_acc_weights = [0.0 for _ in weights]
        self.grad_acc_bias = 0.0

    def forward(self, inputList):
        self.last_input = inputList
        z = sum(i * w for i, w in zip(inputList, self.weights)) + self.bias
        self.last_output = self.relu(z)
        return self.last_output

    def relu(self, x):
        return x if x > 0 else 0.01 * x  # Leaky ReLU

    def relu_derivative(self):
        return 1 if self.last_output > 0 else 0.01

    def delta_backward(self, target):
        return (self.last_output - target) * self.relu_derivative()

    def accumulate_gradients(self):
        for i in range(len(self.weights)):
            self.grad_acc_weights[i] += self.delta * self.last_input[i]
        self.grad_acc_bias += self.delta

    def update_weights(self, learning_rate, batch_size):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * (self.grad_acc_weights[i] / batch_size)
            self.grad_acc_weights[i] = 0.0  
        self.bias -= learning_rate * (self.grad_acc_bias / batch_size)
        self.grad_acc_bias = 0.0  
