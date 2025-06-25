from utils import *
from network import network
from training import train
from persistence import *
from simple_tokenizer import charToValueDictionary, valueToCharDictionary
import os

log_path = "log.txt"
model_path = "model.json"
epochs = 75
deeptrain_epochs = 150
max_response_length = 100
batch_size = 4

# Carica o inizializza il modello
if os.path.exists(model_path):
    Layers, charToValueDictionary, valueToCharDictionary = load_full_model(model_path)
    validate_vocab(Layers, valueToCharDictionary)
else:
    fake_input_sequence = [[0.0] for _ in range(6)]  # stessa lunghezza di window_size
    output, Layers = network(fake_input_sequence)

print("Modalità chatbot attiva. Scrivi un messaggio o 'exit' per uscire.")

while True:
    user_input = input("Tu: ").strip()

    if user_input.lower() == "exit":
        save_full_model(Layers, charToValueDictionary, valueToCharDictionary)
        print("Modello salvato. Uscita.")
        break

    elif user_input.lower() == "deeptrain":
        with open(log_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        print("Deep training in corso su tutto il log.txt...")

        train(full_text, charToValueDictionary, valueToCharDictionary, deeptrain_epochs, batch_size, existing_layers=Layers)
        save_full_model(Layers, charToValueDictionary, valueToCharDictionary)

        # Genera una preview dal prompt "cia" per monitorare l'apprendimento
        preview = predict_n_chars("Giove", Layers, charToValueDictionary, valueToCharDictionary, max_response_length)
        print(f"[Preview dopo deeptrain] {preview}")
        continue  # salta log e training standard

    # Genera risposta prima del training
    response = predict_n_chars(user_input, Layers, charToValueDictionary, valueToCharDictionary, max_response_length)
    print(f"Bot: {response}")

    # Salva input nel log
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(user_input + "\n")