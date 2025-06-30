from layer import Layer
from rnn_layer import RNNLayer
from simple_tokenizer import VOCAB_CHARS
from tuning import NEURONS_PER_LAYER, HIDDEN_LAYERS, LEARNING_RATE

# Parametri di rete
neuronsPerLayer = NEURONS_PER_LAYER           # Numero di neuroni per layer denso
HiddenLayers = HIDDEN_LAYERS               # Numero totale di layer (1 RNN + 2 densi)
learning_rate = LEARNING_RATE           # Tasso di apprendimento globale

# Dimensione dell'input = dimensione del vocabolario (one-hot)
input_size = len(VOCAB_CHARS)

# Costruisce ed esegue una rete completa
def network(input_sequence, preloaded_layers=None):
    # Permette di riutilizzare una rete già inizializzata (es. per training)
    if preloaded_layers is not None:
        output = input_sequence
        for layer in preloaded_layers:
            if hasattr(layer, "cell"):     # Se è un RNNLayer
                output = layer.forward(output)
                output = output[-1]        # Usa solo l'ultimo output della sequenza
            else:
                output = layer.forward(output)
        return output, preloaded_layers

    Layers = []

    # Crea il layer RNN iniziale
    rnn_layer = RNNLayer(input_size, neuronsPerLayer, neuronsPerLayer, learning_rate=learning_rate)
    rnn_output_sequence = rnn_layer.forward(input_sequence)
    Layers.append(rnn_layer)

    # Prende l'ultimo stato dell'output della sequenza
    hidden_output = rnn_output_sequence[-1]

    # Crea gli hidden layer densi
    for _ in range(HiddenLayers - 1):
        dense_layer = Layer(hidden_output, neuronsPerLayer, learning_rate=learning_rate)
        Layers.append(dense_layer)
        hidden_output = dense_layer.output

    # Crea il layer di output finale
    final_layer = Layer(hidden_output, len(VOCAB_CHARS), learning_rate=learning_rate)
    Layers.append(final_layer)

    return final_layer.output, Layers
