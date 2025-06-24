from utils import *
from network import network
from random import shuffle
from tqdm import trange
from persistence import save_full_model

def train(text, char_to_id, id_to_char, epochs=1, batch_size=8, existing_layers=None):
    dataset = make_dataset(text, char_to_id)
    log_file = open("training_log.txt", "a", encoding="utf-8")

    for epoch in trange(epochs, desc="🧠 Training", unit="ep"):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            losses = []
            all_layers = []

            for input, target in batch:
                input_floats = [float(i) for i in input]
                output, layers = network(input_floats, existing_layers)
                loss = cross_entropy_loss(softmax(output), target)
                layers[-1].backward(target)
                losses.append(loss)
                all_layers.append(layers)

            for depth in range(len(all_layers[0]) - 2, -1, -1):
                for layers in all_layers:
                    layers[depth].backward_from_next_layer(layers[depth + 1])

            for layers in all_layers:
                for layer in layers:
                    layer.update_weights()
        
        #Log di test per vedere come si sta "evolvendo" la rete alla fine di ogni Epoch. Disabilitare puó rendere la console piú pulita
        response = predict_n_chars("Giove", existing_layers, char_to_id, id_to_char, 15)
        epoch_loss = sum(losses) / len(losses)
        print(f" Epoch: {epoch} || Output: {response} || Epoch Loss: {epoch_loss:.4f}")
        log_file.write(f"Epoch: {epoch} || Output: {response} || Epoch Loss: {epoch_loss:.4f}\n")

        # Aggiorniamo il model in modo definitivo ogni 10 epoch
        if epoch % 10 == 0 and existing_layers != None:
            save_full_model(existing_layers, char_to_id, id_to_char)
            print(f" Model aggiornato all' epoch {epoch}")

    average_loss = sum(losses) / len(losses)
    print(f"\n📉 Loss media su {len(losses)} esempi: {average_loss:.4f}")
    log_file.close()
