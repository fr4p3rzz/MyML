import math

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