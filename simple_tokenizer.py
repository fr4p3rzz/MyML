from utils import *

lowercase = list("abcdefghijklmnopqrstuvwxyz")
uppercase = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
digits = list("0123456789")
punctuation = list(".,!?;:-_()[]{}'\"@#€%^&*/+=<>\\|«»’‘—–")
accents = list("àáâãäåāăąèéêëēĕėęěìíîïīĭįòóôõöøōőùúûüūŭůýÿŷçñÀÁÂÃÄÅĀĂĄÈÉÊËĒĔĖĘĚÌÍÎÏĪĬĮÒÓÔÕÖØŌŐÙÚÛÜŪŬŮÝŸŶÇÑ")
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

