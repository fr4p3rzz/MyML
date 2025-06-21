from layer import Layer
from simple_tokenizer import VOCAB_CHARS

neuronsPerLayer = 5
HiddenLayers = 2

def network(inputList):
    print("Input iniziale:", inputList)

    # Primo layer
    layer0 = Layer(inputList, neuronsPerLayer)
    hidden_output = layer0.output

    print("Output del layer originale:", hidden_output, "\n\n")

    # Hidden layers
    hidden_layers = []
    for i in range(HiddenLayers):
        hidden_layer = Layer(hidden_output, neuronsPerLayer)
        hidden_layers.append(hidden_layer)
        hidden_output = hidden_layer.output
        print(f"Output del layer nascosto {i}:", hidden_output, "\n\n")

    # Output layer
    final_layer = Layer(hidden_output, len(VOCAB_CHARS))
    print("Output finale:", final_layer.output, "\n\n")

    return final_layer.output