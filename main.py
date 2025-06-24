from utils import *
from network import network
from training import train
from persistence import save_model, load_model
from simple_tokenizer import charToValueDictionary, valueToCharDictionary
import os

log_path = "log.txt"
model_path = "model.json"
epochs = 75
deeptrain_epochs = 100
max_response_length = 100

# Carica o inizializza il modello
if os.path.exists(model_path):
    Layers = load_model(model_path)
else:
    output, Layers = network([0.0] * len(charToValueDictionary)) 

print("Modalità chatbot attiva. Scrivi un messaggio o 'exit' per uscire.")

while True:
    user_input = input("Tu: ").strip()

    if user_input.lower() == "exit":
        save_model(Layers)
        print("Modello salvato. Uscita.")
        break

    elif user_input.lower() == "deeptrain":
        with open(log_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        print("Deep training in corso su tutto il log.txt...")

        train(full_text, charToValueDictionary, valueToCharDictionary, deeptrain_epochs, 8)
        save_model(Layers)

        # Genera una preview dal prompt "cia" per monitorare l'apprendimento
        preview = predict_n_chars("cia", Layers, charToValueDictionary, valueToCharDictionary, max_response_length)
        print(f"[Preview dopo deeptrain] {preview}")
        continue  # salta log e training standard

    # Genera risposta prima del training
    response = predict_n_chars(user_input, Layers, charToValueDictionary, valueToCharDictionary, max_response_length)
    print(f"Bot: {response}")

    # Salva input nel log
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(user_input + "\n")