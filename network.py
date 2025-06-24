from layer import Layer
from simple_tokenizer import VOCAB_CHARS

neuronsPerLayer = 12
HiddenLayers = 5

def network(inputList, preloaded_layers=None):
    if preloaded_layers is not None:
        # usa i layer esistenti
        output = inputList
        for layer in preloaded_layers:
            output = layer.forward(output)
        return output, preloaded_layers
    
    Layers = []

    # Primo layer
    layer0 = Layer(inputList, neuronsPerLayer)
    Layers.append(layer0)
    hidden_output = layer0.output

    # Hidden layers
    hidden_layers = []
    for i in range(HiddenLayers):
        hidden_layer = Layer(hidden_output, neuronsPerLayer)
        hidden_layers.append(hidden_layer)
        hidden_output = hidden_layer.output
        Layers.append(hidden_layer)


    # Output layer
    final_layer = Layer(hidden_output, len(VOCAB_CHARS))
    Layers.append(final_layer)


    return final_layer.output, Layers