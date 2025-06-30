
# Centro di controllo iperparametri
# Modificare questi parametri modificherá le capacitá della RNN

################################################################################################################################
# ATTENZIONE: modificare questi parametri invaliderá l' attuale MODEL.json, rendendo necessario ricominciare da 0 il training! #
# Sará necessario, dopo aver aggiornato questi numeri, eliminare manualmente model.json.                                       #
################################################################################################################################

# Dimensione della finestra di contesto (es. 25 caratteri)
WINDOW_SIZE = 25

# Numero di neuroni per ogni hidden layer
NEURONS_PER_LAYER = 64

# Numero totale di layer (1 RNN + n-1 dense)
HIDDEN_LAYERS = 3

# Tasso di apprendimento
LEARNING_RATE = 0.01

# Numero massimo di caratteri generati in output
MAX_RESPONSE_LENGTH = 100

# Dimensione dei batch per l’addestramento
BATCH_SIZE = 6

# Epoche di deep training
DEEPTRAIN_EPOCHS = 390

# Epoche di training standard (Legacy: attualmente non usato attivamente)
DEFAULT_EPOCHS = 75

# Prompt fisso per generazioni di test
EVAL_PROMPT = "Giove"
