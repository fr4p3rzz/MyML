from utils import *

lowercase = list("abcdefghijklmnopqrstuvwxyz")
uppercase = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
digits = list("0123456789")
punctuation = list(".,!?;:-_()[]{}'\"@#€%^&*/+=<>\\|«»’‘—–")
accents = list("àáâãäåāăąèéêëēĕėěìíîïīĭòóôõöōőùúûüūŭůÀÁÂÃÄĀĂÈÉÊËĒĔĖĘĚÌÍÎÏĪĬĮÒÓÔÖØŌŐÙÚÛÜŪŬŮ")
specials = [' ', '\t', '\n']

VOCAB_CHARS = lowercase + uppercase + digits + punctuation + accents + specials
charToValueDictionary = {c: i for i, c in enumerate(VOCAB_CHARS)}
valueToCharDictionary = {i: c for i, c in enumerate(VOCAB_CHARS)}
    
def getValueFromChar(char):
    if char in charToValueDictionary:
        return charToValueDictionary[char]
    return None

def getCharFromValue(value):
    if value in valueToCharDictionary:
        return valueToCharDictionary[value]
    return None

def make_one_hot_sequence(input_ids, vocab_size):
    return [
        [1.0 if j == idx else 0.0 for j in range(vocab_size)]
        for idx in input_ids
    ]
