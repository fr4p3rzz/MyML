import json
from layer import Layer
from neuron import Neuron

def save_model(layers, filename="model.json"):
    serializable = []
    for layer in layers:
        layer_data = []
        for neuron in layer.neurons:
            neuron_data = {
                "weights": neuron.weights,
                "bias": neuron.bias
            }
            layer_data.append(neuron_data)
        serializable.append(layer_data)
    
    with open(filename, "w") as f:
        json.dump(serializable, f)
    print(f"Modello salvato su {filename}")

def load_model(filename="model.json"):
    import json
    with open(filename, "r") as f:
        data = json.load(f)

    layers = []
    for layer_data in data:
        neurons = []
        for neuron_data in layer_data:
            neuron = Neuron(
                weights=neuron_data["weights"],
                bias=neuron_data["bias"]
            )
            neurons.append(neuron)

        dummy_input = [0.0] * len(neurons[0].weights)
        layer = Layer(dummy_input, len(neurons))
        layer.neurons = neurons
        layers.append(layer)
    
    print(f"Modello caricato da {filename}")
    return layers