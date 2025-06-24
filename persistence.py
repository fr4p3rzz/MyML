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

def save_full_model(layers, char_to_id, id_to_char, filename="model.json"):
    serializable = {
        "layers": [],
        "char_to_id": char_to_id,
        "id_to_char": id_to_char
    }

    for layer in layers:
        layer_data = []
        for neuron in layer.neurons:
            neuron_data = {
                "weights": neuron.weights,
                "bias": neuron.bias
            }
            layer_data.append(neuron_data)
        serializable["layers"].append(layer_data)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"✅ Modello completo salvato su {filename}")

def load_full_model(filename="model.json"):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers_data = data["layers"]
    char_to_id = data["char_to_id"]
    id_to_char = {int(k): v for k, v in data["id_to_char"].items()}  # Fix per chiavi numeriche

    layers = []
    for layer_data in layers_data:
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

    print(f"✅ Modello completo caricato da {filename}")
    return layers, char_to_id, id_to_char