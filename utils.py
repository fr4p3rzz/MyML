import math
from network import network
from simple_tokenizer import make_one_hot_sequence

def softmax(values: list[float]) -> list[float]:

    dividend = [math.exp(i) for i in values]
    divisor = sum(dividend)

    results = [i / divisor for i in dividend]


    return results

def make_dataset(text: str, char_to_id: dict, window_size: int = 6):
    encoded = encode(text, char_to_id)
    dataset = []

    for i in range(len(encoded) - window_size):
        input_seq = encoded[i : i + window_size]        # es. [12, 4, 0, 15]
        target = encoded[i + window_size]               # es. 8
        dataset.append((input_seq, target))

    return dataset

def cross_entropy_loss(predicted_probs, target_vector):
    return -sum(t * math.log(p + 1e-12) for p, t in zip(predicted_probs, target_vector))


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

def predict_n_chars(prompt, layers, char_to_id, id_to_char, max_length=100, window_size=6):
    generated = prompt

    for _ in range(max_length):
        
        input_seq = generated[-window_size:]

        try:
            input_ids = [char_to_id[c] for c in input_seq]
        except KeyError:
            break  # carattere ignoto

        # Padding se troppo corta
        while len(input_ids) < window_size:
            input_ids.insert(0, 0)

        # One-hot sequence
        input_sequence = make_one_hot_sequence(input_ids, len(char_to_id))

        # Forward pass
        output, _ = network(input_sequence, layers)
        probs = softmax(output)

        # Scegli il carattere con probabilità massima
        next_id = probs.index(max(probs))
        next_char = id_to_char[next_id]
        generated += next_char

    return generated[len(prompt):]



def validate_vocab(model, id_to_char):
    output_size = len(model[-1].neurons)
    vocab_size = len(id_to_char)
    if output_size != vocab_size:
        raise ValueError(f"Mismatch tra output del modello ({output_size}) e vocab size ({vocab_size})")