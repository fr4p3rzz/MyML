import math
from network import network

def softmax(values: list[float]) -> list[float]:

    dividend = [math.exp(i) for i in values]
    divisor = sum(dividend)

    results = [i / divisor for i in dividend]


    return results

def make_dataset(text: str, char_to_id: dict, window_size: int = 4):
    encoded = encode(text, char_to_id)
    dataset = []

    for i in range(len(encoded) - window_size):
        input_seq = encoded[i : i + window_size]        # es. [12, 4, 0, 15]
        target = encoded[i + window_size]               # es. 8
        dataset.append((input_seq, target))

    return dataset

def cross_entropy_loss(pred: list[float], target: int) -> float:
    epsilon = 1e-12
    if target >= len(pred):
        raise ValueError(f"Target index {target} is out of bounds for prediction of size {len(pred)}")

    return - math.log(pred[target] + epsilon) 


def encode(str, char_to_id) -> list[int]:
    output = []
    for c in str:
        output.append(char_to_id[c])

    return output

def decode(code_list: list[int], id_to_char: dict[int, str]) -> str:
    string = ""
    for i in code_list:
        string += id_to_char[i]

    return string

def predict(prompt, layers, char_to_id, id_to_char, max_len=100):
    generated = prompt

    for _ in range(max_len):
        window = generated[-4:]  # ultimi 4 caratteri
        input_encoded = [float(char_to_id.get(c, 0)) for c in window]
        output, _ = network(input_encoded)
        probs = softmax(output)
        next_id = probs.index(max(probs))  # greedy
        next_char = id_to_char[next_id]
        generated += next_char

        if next_char == '\n':  # oppure altro segnale di stop
            break

    return generated[len(prompt):]

def predict_n_chars(prompt, layers, char_to_id, id_to_char, max_length=100, window_size=4):
    generated = prompt

    for _ in range(max_length):
        input_seq = generated[-window_size:]  # Prende gli ultimi n caratteri
        try:
            input_ids = [float(char_to_id[c]) for c in input_seq]
        except KeyError:
            break  # Carattere non nel vocabolario

        # Padding se troppo corto
        while len(input_ids) < window_size:
            input_ids.insert(0, 0.0)

        output, _ = network(input_ids)
        probs = softmax(output)

        # Scegli carattere con probabilità massima
        next_id = probs.index(max(probs))
        next_char = id_to_char[next_id]

        generated += next_char

    return generated[len(prompt):] 