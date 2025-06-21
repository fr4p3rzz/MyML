from utils import *
from network import network

def train(text, char_to_id, id_to_char):

    dataset = make_dataset(text, char_to_id)
    normalizedLogits = []
    losses = []

    for step, (input, target) in enumerate(dataset):
            input_floats = [float(i) for i in input]
            normalized = softmax(network(input_floats))
            normalizedLogits.append(normalized)
            losses.append(cross_entropy_loss(normalized, target))

            if step < 5:
                print(f"Step {step}: input {input} → target {target}, loss = {losses[-1]:.4f}")


    average_loss = sum(losses) / len(losses)
    print(f"\nLoss media su {len(losses)} esempi: {average_loss:.4f}")