import math
import random

def tanh(x):
    return [math.tanh(v) for v in x]

def dot(A, B):  # Matrice A * Vettore B
    result = []
    for row in A:
        result.append(sum(a * b for a, b in zip(row, B)))
    return result

def add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

class RNNCell:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        scale = 1.0 / math.sqrt(input_size)  # Xavier initialization per tanh
        self.W_xh = [[random.uniform(-scale, scale) for _ in range(input_size)] for _ in range(hidden_size)]
        self.W_hh = [[random.uniform(-scale, scale) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.b_h =  [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]

        self.W_hy = self.W_hy = [[random.uniform(-scale, scale) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b_y = [0.0 for _ in range(output_size)]

        self.h = [0.0 for _ in range(hidden_size)]

    def forward(self, x, h_prev):
        h_candidate = add(dot(self.W_xh, x), dot(self.W_hh, h_prev))
        h_next = tanh(add(h_candidate, self.b_h))
        y = add(dot(self.W_hy, h_next), self.b_y)
        return y, h_next