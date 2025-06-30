from neuron import Neuron
import random

# La classe Layer rappresenta un singolo strato di una rete neurale feedforward.
class Layer:
    def __init__(self, inputList, activeNeurons, learning_rate=0.01):
        # Inizializza il vettore di input fornito
        self.inputList = inputList

        # Imposta il tasso di apprendimento per la retropropagazione
        self.learning_rate = learning_rate

        # Crea una lista di neuroni con pesi e bias inizializzati casualmente
        self.neurons = [
            Neuron(
                setupWeights(1, len(inputList), -0.03, 0.03)[0],  # inizializza i pesi con valori casuali tra -0.03 e 0.03
                setupBiases(1)[0]  # inizializza il bias con un valore casuale tra 0.5 e 1.5
            )
            for _ in range(activeNeurons)
        ]

        # Calcola l'output iniziale per ogni neurone del layer
        self.output = [n.forward(inputList) for n in self.neurons]

    # Esegue il forward pass: aggiorna l'input e calcola i nuovi output
    def forward(self, inputList):
        self.inputList = inputList
        self.output = [n.forward(inputList) for n in self.neurons]
        return self.output

    # Calcola i delta per l'output layer rispetto al target desiderato (one-hot)
    def backward(self, targetIndex):
        # Crea un vettore target con 1 nella posizione corretta, 0 altrove
        target_vector = [1 if i == targetIndex else 0 for i in range(len(self.neurons))]

        # Calcola la delta per ciascun neurone in base al target
        for neuron, target in zip(self.neurons, target_vector):
            neuron.delta = neuron.delta_backward(target)

    # Calcola i delta per un layer nascosto sulla base del layer successivo
    def backward_from_next_layer(self, next_layer):
        for i, neuron in enumerate(self.neurons):
            downstream_gradient = 0.0

            # Somma i gradienti pesati provenienti dal layer successivo
            for next_neuron in next_layer.neurons:
                if i < len(next_neuron.weights):
                    downstream_gradient += next_neuron.weights[i] * next_neuron.delta

            # Applica la derivata della ReLU per ottenere la delta del neurone
            neuron.delta = downstream_gradient * neuron.relu_derivative()

    # Accumula i gradienti per ciascun neurone (utilizzato nel batch training)
    def accumulate_gradients(self):
        for neuron in self.neurons:
            neuron.accumulate_gradients()

    # Applica l'aggiornamento dei pesi usando i gradienti medi del batch
    def update_weights(self, batch_size):
        for neuron in self.neurons:
            neuron.update_weights(self.learning_rate, batch_size)

# Inizializza una lista di pesi casuali per un certo numero di neuroni
def setupWeights(neuronsToInitialize, requestedWeights, minValue, maxValue):
    weightList = []

    # Per ogni neurone, crea un vettore di pesi casuali
    for n in range(neuronsToInitialize):
        weights = [random.uniform(minValue, maxValue) for i in range(requestedWeights)]
        weightList.append(weights)

    return weightList

# Inizializza una lista di bias casuali per ciascun neurone
def setupBiases(activeNeurons):
    return [random.uniform(0.5, 1.5) for _ in range(activeNeurons)]
