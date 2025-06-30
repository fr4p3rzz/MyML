import math

# Neurone base per una rete fully connected con attivazione Leaky ReLU
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights              # Pesi del neurone
        self.bias = bias                    # Bias del neurone
        self.last_input = None              # Input salvato per il backprop
        self.last_output = None            # Output salvato per il backprop
        self.delta = None                   # Gradiente locale dell’errore

        # Accumulatori per i gradienti (batch training)
        self.grad_acc_weights = [0.0 for _ in weights]
        self.grad_acc_bias = 0.0

    # Forward pass: calcola z = wx + b, poi attiva con Leaky ReLU
    def forward(self, inputList):
        self.last_input = inputList
        z = sum(i * w for i, w in zip(inputList, self.weights)) + self.bias
        self.last_output = self.relu(z)
        return self.last_output

    # Funzione di attivazione: Leaky ReLU
    def relu(self, x):
        return x if x > 0 else 0.01 * x

    # Derivata di Leaky ReLU in base all’output calcolato
    def relu_derivative(self):
        return 1 if self.last_output > 0 else 0.01

    # Calcolo del delta rispetto al target (solo per output layer)
    def delta_backward(self, target):
        return (self.last_output - target) * self.relu_derivative()

    # Accumula i gradienti rispetto ai pesi e al bias
    def accumulate_gradients(self):
        for i in range(len(self.weights)):
            self.grad_acc_weights[i] += self.delta * self.last_input[i]
        self.grad_acc_bias += self.delta

    # Applica l'aggiornamento pesi con media dei gradienti nel batch
    def update_weights(self, learning_rate, batch_size):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * (self.grad_acc_weights[i] / batch_size)
            self.grad_acc_weights[i] = 0.0  # reset
        self.bias -= learning_rate * (self.grad_acc_bias / batch_size)
        self.grad_acc_bias = 0.0  # reset
