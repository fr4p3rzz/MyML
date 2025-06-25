
import json
from layer import Layer
from neuron import Neuron
from rnn_layer import RNNLayer
from rnn_cell import RNNCell

def save_full_model(layers, char_to_id, id_to_char, filename="model.json"):
    serializable = {
        "layers": [],
        "char_to_id": char_to_id,
        "id_to_char": id_to_char
    }

    for layer in layers:
        if isinstance(layer, Layer):
            layer_data = {
                "type": "Layer",
                "neurons": [
                    {"weights": neuron.weights, "bias": neuron.bias}
                    for neuron in layer.neurons
                ]
            }
        elif isinstance(layer, RNNLayer):
            cell = layer.cell
            layer_data = {
                "type": "RNNLayer",
                "input_size": len(cell.W_xh[0]),
                "hidden_size": cell.hidden_size,
                "output_size": len(cell.W_hy),
                "weights": {
                    "W_xh": cell.W_xh,
                    "W_hh": cell.W_hh,
                    "W_hy": cell.W_hy,
                    "b_h": cell.b_h,
                    "b_y": cell.b_y
                }
            }
        else:
            raise ValueError("Unknown layer type")
        serializable["layers"].append(layer_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"✅ Modello completo salvato su {filename}")

def load_full_model(filename="model.json"):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers_data = data["layers"]
    char_to_id = data["char_to_id"]
    id_to_char = {int(k): v for k, v in data["id_to_char"].items()}

    layers = []
    for layer_data in layers_data:
        if layer_data["type"] == "Layer":
            neurons = []
            for neuron_data in layer_data["neurons"]:
                neuron = Neuron(
                    weights=neuron_data["weights"],
                    bias=neuron_data["bias"]
                )
                neurons.append(neuron)
            dummy_input = [0.0] * len(neurons[0].weights)
            layer = Layer(dummy_input, len(neurons))
            layer.neurons = neurons
            layers.append(layer)

        elif layer_data["type"] == "RNNLayer":
            weights = layer_data["weights"]
            cell = RNNCell(
                input_size=layer_data["input_size"],
                hidden_size=layer_data["hidden_size"],
                output_size=layer_data["output_size"]
            )
            cell.W_xh = weights["W_xh"]
            cell.W_hh = weights["W_hh"]
            cell.W_hy = weights["W_hy"]
            cell.b_h = weights["b_h"]
            cell.b_y = weights["b_y"]
            layer = RNNLayer(layer_data["input_size"], layer_data["hidden_size"], layer_data["output_size"])
            layer.cell = cell
            layers.append(layer)

        else:
            raise ValueError("Unknown layer type in file")

    print(f"✅ Modello completo caricato da {filename}")
    return layers, char_to_id, id_to_char
