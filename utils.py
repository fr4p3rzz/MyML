import math
from network import network
from simple_tokenizer import make_one_hot_sequence
from tuning import WINDOW_SIZE
# Applica la softmax a una lista di valori
def softmax(values: list[float]) -> list[float]:
    dividend = [math.exp(i) for i in values]      # Esponenzia ogni valore
    divisor = sum(dividend)                       # Somma per normalizzazione
    results = [i / divisor for i in dividend]     # Probabilità normalizzate
    return results

# Crea un dataset (input, target) da una stringa, con window sliding
def make_dataset(text: str, char_to_id: dict, window_size: int = WINDOW_SIZE):
    encoded = encode(text, char_to_id)            # Converte testo in indici
    dataset = []
    for i in range(len(encoded) - window_size):
        input_seq = encoded[i : i + window_size]  # Sequenza di input
        target = encoded[i + window_size]         # Target successivo
        dataset.append((input_seq, target))
    return dataset

# Calcola la cross entropy tra le probabilità e il vettore target
def cross_entropy_loss(predicted_probs, target_vector):
    return -sum(t * math.log(p + 1e-12) for p, t in zip(predicted_probs, target_vector))

# Codifica una stringa in una lista di indici
def encode(str, char_to_id) -> list[int]:
    output = []
    for c in str:
        output.append(char_to_id[c])
    return output

# Decodifica una lista di indici in una stringa
def decode(code_list: list[int], id_to_char: dict[int, str]) -> str:
    string = ""
    for i in code_list:
        string += id_to_char[i]
    return string

# Genera una sequenza di testo dato un prompt iniziale
def predict_n_chars(prompt, layers, char_to_id, id_to_char, max_length=100, window_size=WINDOW_SIZE):
    generated = prompt

    for _ in range(max_length):
        input_seq = generated[-window_size:]  # Ultimi caratteri noti

        try:
            input_ids = [char_to_id[c] for c in input_seq]
        except KeyError:
            break  # Interrompi se contiene caratteri sconosciuti

        # Padding se input troppo corto
        while len(input_ids) < window_size:
            input_ids.insert(0, 0)

        input_sequence = make_one_hot_sequence(input_ids, len(char_to_id))  # One-hot
        output, _ = network(input_sequence, layers)                         # Forward
        probs = softmax(output)                                             # Probabilità
        next_id = probs.index(max(probs))                                   # Scelta deterministica
        next_char = id_to_char[next_id]
        generated += next_char

    return generated[len(prompt):]  # Rimuove il prompt dalla risposta

# Controlla che il numero di neuroni in output coincida con il vocabolario
def validate_vocab(model, id_to_char):
    output_size = len(model[-1].neurons)
    vocab_size = len(id_to_char)
    if output_size != vocab_size:
        raise ValueError(f"Mismatch tra output del modello ({output_size}) e vocab size ({vocab_size})")
