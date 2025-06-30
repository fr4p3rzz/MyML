from utils import *

# Caratteri base del vocabolario
lowercase = list("abcdefghijklmnopqrstuvwxyz")
uppercase = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
digits = list("0123456789")
punctuation = list(".,!?;:-_()[]{}'\"@#€%^&*/+=<>\\|«»’‘—–")
accents = list("àáâãäåāăąèéêëēĕėěìíîïīĭòóôõöōőùúûüūŭůÀÁÂÃÄĀĂÈÉÊËĒĔĖĘĚÌÍÎÏĪĬĮÒÓÔÖØŌŐÙÚÛÜŪŬŮ")
specials = [' ', '\t', '\n']  # Spazi, tabulazioni e newline

# Lista finale di caratteri supportati dal modello
VOCAB_CHARS = lowercase + uppercase + digits + punctuation + accents + specials

# Dizionario: carattere → indice
charToValueDictionary = {c: i for i, c in enumerate(VOCAB_CHARS)}

# Dizionario: indice → carattere
valueToCharDictionary = {i: c for i, c in enumerate(VOCAB_CHARS)}

# Restituisce l'indice associato a un carattere, oppure None se non trovato
def getValueFromChar(char):
    if char in charToValueDictionary:
        return charToValueDictionary[char]
    return None

# Restituisce il carattere associato a un indice, oppure None se non valido
def getCharFromValue(value):
    if value in valueToCharDictionary:
        return valueToCharDictionary[value]
    return None

# Converte una sequenza di indici in one-hot encoding
def make_one_hot_sequence(input_ids, vocab_size):
    return [
        [1.0 if j == idx else 0.0 for j in range(vocab_size)]
        for idx in input_ids
    ]
