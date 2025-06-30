# Diario di bordo (auto-documentato):
# Transizione da encoding float a one-hot:
# Prima ogni carattere veniva rappresentato da un singolo float normalizzato: index / vocabolario.
# → Questo approccio era semplice ma poco informativo: la rete imparava dipendenze artificiali (es. che 'a' e 'b' sono vicini).
# → Funzionava solo perché il modello iniziale era una rete feedforward senza memoria, quindi imparava solo corrispondenze singole.
# Con l'introduzione della RNN, è più utile usare un encoding one-hot:
# → Ogni carattere diventa un vettore binario lungo come l'intero dizionario dove solo l'indice corretto è 1, gli altri sono 0.
# → Questo rimuove qualsiasi assunzione di ordine e rende più chiaro l'apprendimento.
# Poco ottimizzato data la dimensione dei vettori e meno efficace dell'embedding, ma ottimo per fini didattici e per gestire una piccola rete home-made tenendo la codebase minimale.

from utils import *
from simple_tokenizer import make_one_hot_sequence, VOCAB_CHARS
from network import network
from tqdm import trange
from persistence import save_full_model
from rnn_layer import RNNLayer
import time

# Ciclo di training principale
def train(text, char_to_id, id_to_char, epochs=1, batch_size=8, existing_layers=None):
    dataset = make_dataset(text, char_to_id)  # Crea dataset in formato sliding window
    log_file = open("training_log.txt", "a", encoding="utf-8")
    total_start_time = time.time()

    # Ciclo per epoche
    for epoch in trange(epochs, desc="🧠 Training", unit="ep"):
        epoch_start = time.time()
        losses = []

        # Batch training
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]

            # Per ogni esempio nel batch
            for input, target in batch:
                input_sequence = make_one_hot_sequence(input, len(char_to_id))
                output, _ = network(input_sequence, existing_layers)
                probs = softmax(output)

                # Costruisci vettore target one-hot
                target_vector = [0.0] * len(char_to_id)
                target_vector[target] = 1.0

                # Calcola la loss e salva
                loss = cross_entropy_loss(probs, target_vector)
                losses.append(loss)

                # Backward su output layer
                existing_layers[-1].backward(target)

                # Backward su hidden layers (densi e RNN)
                for depth in range(len(existing_layers) - 2, -1, -1):
                    current_layer = existing_layers[depth]
                    next_layer = existing_layers[depth + 1]
                    if isinstance(current_layer, RNNLayer):
                        dY_sequence = [[0.0 for _ in out] for out in current_layer.output]
                        for j, dy in enumerate(next_layer.neurons):
                            for i, h in enumerate(current_layer.hidden_states[-1]):
                                dY_sequence[-1][j] += dy.weights[i] * dy.delta
                        current_layer.backward(dY_sequence)
                    else:
                        current_layer.backward_from_next_layer(next_layer)
                
                # Accumula gradienti per il batch
                for layer in existing_layers:
                    if hasattr(layer, "accumulate_gradients"):
                        layer.accumulate_gradients()

            # Aggiornamento pesi alla fine del batch
            for layer in existing_layers:
                if hasattr(layer, "update_weights"):
                    layer.update_weights(batch_size=len(batch))

        # Generazione di esempio alla fine dell’epoca
        response = predict_n_chars("Giove", existing_layers, char_to_id, id_to_char, 60)

        # Log e stampa
        epoch_loss = sum(losses) / len(losses)
        log_file.write(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Output: '{response}'\n")
        print(f" Epoch {epoch} | Loss: {epoch_loss:.4f}| Output: '{response}'")

        # Salvataggio periodico del modello
        if epoch % 5 == 0:
            save_full_model(existing_layers, char_to_id, id_to_char)
            print(f"Modello salvato all'epoch {epoch}")

    # Report finale
    total_time = time.time() - total_start_time
    average_loss = sum(losses) / len(losses)
    print(f"\n✅ Deeptrain completato in {total_time/60:.1f} minuti")
    print(f"📉 Loss media su {len(losses)} esempi: {average_loss:.4f}")
    log_file.close()
