from layer import Layer
from rnn_layer import RNNLayer
from simple_tokenizer import VOCAB_CHARS

neuronsPerLayer = 64
HiddenLayers = 2
learning_rate = 0.05

# Ora input_size coincide con la dimensione del vocabolario, perché usiamo encoding one-hot
input_size = len(VOCAB_CHARS)

def network(input_sequence, preloaded_layers=None):
    if preloaded_layers is not None:
        output = input_sequence
        for layer in preloaded_layers:
            if hasattr(layer, "cell"):
                output = layer.forward(output)
                output = output[-1]
            else:
                output = layer.forward(output)
        return output, preloaded_layers

    Layers = []
    rnn_layer = RNNLayer(input_size, neuronsPerLayer, neuronsPerLayer, learning_rate=learning_rate)
    rnn_output_sequence = rnn_layer.forward(input_sequence)
    Layers.append(rnn_layer)
    hidden_output = rnn_output_sequence[-1]

    for _ in range(HiddenLayers - 1):
        dense_layer = Layer(hidden_output, neuronsPerLayer, learning_rate=learning_rate)
        Layers.append(dense_layer)
        hidden_output = dense_layer.output

    final_layer = Layer(hidden_output, len(VOCAB_CHARS), learning_rate=learning_rate)
    Layers.append(final_layer)

    return final_layer.output, Layers
