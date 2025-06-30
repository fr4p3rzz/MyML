import math
import random

# Applica tanh elemento per elemento su un vettore
def tanh(x):
    return [math.tanh(v) for v in x]

# Prodotto matrice-vettore
def dot(A, B):
    result = []
    for row in A:
        result.append(sum(a * b for a, b in zip(row, B)))
    return result

# Somma elemento per elemento di due vettori
def add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

# Cella base RNN: calcola uno step temporale su input e stato precedente
class RNNCell:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        # Xavier initialization per pesi legati a tanh
        scale = 1.0 / math.sqrt(input_size)

        # Pesi input -> hidden
        self.W_xh = [[random.uniform(-scale, scale) for _ in range(input_size)]
                     for _ in range(hidden_size)]

        # Pesi hidden -> hidden (ricorrenza)
        self.W_hh = [[random.uniform(-scale, scale) for _ in range(hidden_size)]
                     for _ in range(hidden_size)]

        # Bias per hidden
        self.b_h = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]

        # Pesi hidden -> output
        self.W_hy = [[random.uniform(-scale, scale) for _ in range(hidden_size)]
                     for _ in range(output_size)]

        # Bias output
        self.b_y = [0.0 for _ in range(output_size)]

        # Stato nascosto corrente (opzionale, utile per RNN stateless/stateful)
        self.h = [0.0 for _ in range(hidden_size)]

    # Esegue un passo della cella RNN
    def forward(self, x, h_prev):
        # Calcola l'input aggregato per lo stato nascosto
        h_candidate = add(dot(self.W_xh, x), dot(self.W_hh, h_prev))

        # Applica la funzione di attivazione tanh
        h_next = tanh(add(h_candidate, self.b_h))

        # Calcola l'output a partire dallo stato nascosto
        y = add(dot(self.W_hy, h_next), self.b_y)

        return y, h_next
