from utils import *
from network import network
from training import train
from persistence import *
from simple_tokenizer import charToValueDictionary, valueToCharDictionary
from tuning import MAX_RESPONSE_LENGTH, BATCH_SIZE, DEEPTRAIN_EPOCHS, EVAL_PROMPT
import os

# Impostazioni generali
log_path = "log.txt"                       # File di log per deep training
model_path = "model.json"                 # Percorso per salvataggio/caricamento modello
deeptrain_epochs = DEEPTRAIN_EPOCHS                   # Epoche per il deep training
max_response_length = MAX_RESPONSE_LENGTH               # Numero massimo di caratteri generati
batch_size = BATCH_SIZE                            # Dimensione del batch per training

# Carica il modello esistente o inizializza uno nuovo
if os.path.exists(model_path):
    Layers, charToValueDictionary, valueToCharDictionary = load_full_model(model_path)
    validate_vocab(Layers, valueToCharDictionary)
else:
    fake_input_sequence = [[0.0] for _ in range(25)]  # input fittizio per inizializzare i layer
    output, Layers = network(fake_input_sequence)

print("Modalità chatbot attiva. Scrivi un messaggio o 'exit' per uscire.")

# Loop principale di interazione utente
while True:
    user_input = input("Tu: ").strip()

    if user_input.lower() == "exit":
        save_full_model(Layers, charToValueDictionary, valueToCharDictionary)
        print("Modello salvato. Uscita.")
        break

    elif user_input.lower() == "deeptrain":
        # Carica tutto il log per fare deep training
        with open(log_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        print("Deep training in corso su tutto il log.txt...")

        # Esegue training prolungato su log pregresso
        train(full_text, charToValueDictionary, valueToCharDictionary,
              deeptrain_epochs, batch_size, existing_layers=Layers)

        save_full_model(Layers, charToValueDictionary, valueToCharDictionary)

        # Genera una preview dal prompt fisso per controllo qualità
        preview = predict_n_chars(EVAL_PROMPT, Layers, charToValueDictionary, valueToCharDictionary, max_response_length)
        print(f"[Preview dopo deeptrain] {preview}")
        continue

    # Genera risposta in base all'input utente
    response = predict_n_chars(user_input, Layers, charToValueDictionary, valueToCharDictionary, max_response_length)
    print(f"Bot: {response}")

    # Salva l'input dell'utente nel log
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(user_input + "\n")
