from rnn_cell import RNNCell

# RNNLayer gestisce una sequenza temporale tramite una cella RNN interna
class RNNLayer:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.cell = RNNCell(input_size, hidden_size, output_size)  # Cella RNN singola condivisa nel tempo
        self.hidden_states = []  # Stati nascosti temporali
        self.inputs = []  # Sequenza di input
        self.output = []  # Sequenza di output
        self.learning_rate = learning_rate  # Tasso di apprendimento

        # Inizializza accumulatori per il batch update
        self.grad_acc_Wxh = [[0.0 for _ in row] for row in self.cell.W_xh]
        self.grad_acc_Whh = [[0.0 for _ in row] for row in self.cell.W_hh]
        self.grad_acc_bh = [0.0 for _ in self.cell.b_h]
        self.grad_acc_Why = [[0.0 for _ in row] for row in self.cell.W_hy]
        self.grad_acc_by = [0.0 for _ in self.cell.b_y]
        self.batch_count = 0  # Conteggio dei campioni nel batch

    # Esegue il forward pass su una sequenza di input
    def forward(self, x_sequence):
        self.hidden_states = []
        self.inputs = []
        self.output = []
        h_prev = [0.0] * self.cell.hidden_size  # Stato nascosto iniziale

        output_sequence = []
        for x_t in x_sequence:
            y_t, h_t = self.cell.forward(x_t, h_prev)  # Step temporale
            self.inputs.append(x_t)
            self.hidden_states.append(h_t)
            self.output.append(y_t)
            output_sequence.append(y_t)
            h_prev = h_t  # Passa lo stato al passo successivo

        return output_sequence

    # Backpropagation through time (BPTT)
    def backward(self, dY_sequence):
        # Inizializza i gradienti locali
        dWxh = [[0.0 for _ in range(len(self.inputs[0]))] for _ in range(self.cell.hidden_size)]
        dWhh = [[0.0 for _ in range(self.cell.hidden_size)] for _ in range(self.cell.hidden_size)]
        dbh = [0.0 for _ in range(self.cell.hidden_size)]
        dWhy = [[0.0 for _ in range(self.cell.hidden_size)] for _ in range(len(dY_sequence[0]))]
        dby = [0.0 for _ in range(len(dY_sequence[0]))]
        dh_next = [0.0 for _ in range(self.cell.hidden_size)]  # Gradiente accumulato da t+1

        # Backward in ordine temporale inverso
        for t in reversed(range(len(self.inputs))):
            x_t = self.inputs[t]
            h_t = self.hidden_states[t]
            h_prev = self.hidden_states[t - 1] if t > 0 else [0.0] * self.cell.hidden_size
            dy = dY_sequence[t]

            # Gradiente output -> Why e by
            for i in range(len(dy)):
                for j in range(len(h_t)):
                    dWhy[i][j] += dy[i] * h_t[j]
                dby[i] += dy[i]

            # Gradiente rispetto all'hidden state
            dh = [sum(self.cell.W_hy[k][i] * dy[k] for k in range(len(dy))) + dh_next[i]
                  for i in range(self.cell.hidden_size)]

            # Derivata tanh
            dt = [dh[i] * (1 - h_t[i] ** 2) for i in range(self.cell.hidden_size)]

            # Gradiente input -> Wxh
            for i in range(self.cell.hidden_size):
                for j in range(len(x_t)):
                    dWxh[i][j] += dt[i] * x_t[j]
                for j in range(self.cell.hidden_size):
                    dWhh[i][j] += dt[i] * h_prev[j]
                dbh[i] += dt[i]

            # Propaga il gradiente allo step precedente
            dh_next = [sum(self.cell.W_hh[j][i] * dt[j] for j in range(self.cell.hidden_size))
                       for i in range(self.cell.hidden_size)]

        # Accumula i gradienti per il batch
        for i in range(self.cell.hidden_size):
            for j in range(len(self.cell.W_xh[i])):
                self.grad_acc_Wxh[i][j] += dWxh[i][j]
            for j in range(len(self.cell.W_hh[i])):
                self.grad_acc_Whh[i][j] += dWhh[i][j]
            self.grad_acc_bh[i] += dbh[i]

        for i in range(len(self.cell.W_hy)):
            for j in range(len(self.cell.W_hy[i])):
                self.grad_acc_Why[i][j] += dWhy[i][j]
            self.grad_acc_by[i] += dby[i]

        self.batch_count += 1  # Conta il batch

    # Applica l'aggiornamento dei pesi con media del batch
    def update_weights(self, batch_size):
        if self.batch_count == 0:
            return  # Evita divisione per zero

        for i in range(self.cell.hidden_size):
            for j in range(len(self.cell.W_xh[i])):
                self.cell.W_xh[i][j] -= self.learning_rate * (self.grad_acc_Wxh[i][j] / self.batch_count)
                self.grad_acc_Wxh[i][j] = 0.0
            for j in range(len(self.cell.W_hh[i])):
                self.cell.W_hh[i][j] -= self.learning_rate * (self.grad_acc_Whh[i][j] / self.batch_count)
                self.grad_acc_Whh[i][j] = 0.0
            self.cell.b_h[i] -= self.learning_rate * (self.grad_acc_bh[i] / self.batch_count)
            self.grad_acc_bh[i] = 0.0

        for i in range(len(self.cell.W_hy)):
            for j in range(len(self.cell.W_hy[i])):
                self.cell.W_hy[i][j] -= self.learning_rate * (self.grad_acc_Why[i][j] / self.batch_count)
                self.grad_acc_Why[i][j] = 0.0
            self.cell.b_y[i] -= self.learning_rate * (self.grad_acc_by[i] / self.batch_count)
            self.grad_acc_by[i] = 0.0

        self.batch_count = 0  # Reset dopo update
