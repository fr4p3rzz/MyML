# Mini Language Model in Python

Questo progetto è un **esperimento a fini didattici** per costruire da zero un semplice modello linguistico (language model) in grado di predire e generare testo, carattere per carattere, senza utilizzare librerie esterne come TensorFlow o PyTorch.
Tutte le componenti (rete neurale, RNN, funzione di attivazione, training loop, salvataggio modello, tokenizer, ecc.) sono implementate manualmente, con lo scopo di capire davvero come funziona un modello linguistico a basso livello.
Questo modello **non é** scalabile, né efficente ed il suo fine NON é quello di creare un modello realmente utilizzabile in real-world situation: serve a prendere dimestichezza con i fondamentali del machine learning e del funzionamento
di un Language Model moderno.

---

## Obiettivi didattici

- Comprendere a fondo le basi delle reti neurali e delle RNN.
- Vedere in pratica come funziona un sistema di generazione del linguaggio.
- Sperimentare con il training autoregressivo e la tokenizzazione.
- Offrire una base semplice e totalmente open-source per chi vuole imparare o insegnare.

---

## Stato attuale

- Recurrent Neural Network composta da:
  - un `RNNLayer` seguita da uno o più layer densi.
- Codifica one-hot su un vocabolario statico configurabile.
- Sistema di salvataggio e caricamento del modello in JSON.
- Modalità interattiva tipo chatbot da terminale.
- Possibilità di effettuare deep training retroattivo su un log riempito con il testo. É possibile aggiungere attivamente al log lo storico delle conversazioni.
- Parametri di tuning completamente configurabili da un file centralizzato.

---

## Funzionamento

 - É necessario avere l' ambiente **Python 3.9** o superiore installato nel proprio sistema
 - Avviare da bash il programma con il comando `python main.py`
 - Il programma caricherá automaticamente il model attuale, se presente.
 - utilizzare la keyword `deeptrain` per iniziare il training intensivo con le epoch definite
 - utilizzare la keyword `exit` per chiudere il programma
 - qualsiasi altro input attiverá la funzione chatbot 

### Parametri e configurazione

Tutti i principali parametri sono definiti nel file `tuning.py`, che funge da centro di controllo del modello:

| Parametro           | Descrizione                                           |
|---------------------|-------------------------------------------------------|
| `WINDOW_SIZE`       | Lunghezza della sequenza di input (numero di caratteri) |
| `NEURONS_PER_LAYER` | Numero di neuroni per ogni layer nascosto             |
| `HIDDEN_LAYERS`     | Numero totale di layer (1 RNN + n-1 densi)            |
| `LEARNING_RATE`     | Tasso di apprendimento                                |
| `BATCH_SIZE`        | Dimensione del batch usato per il training            |
| `DEEPTRAIN_EPOCHS`  | Numero di epoche per il deep training retroattivo     |
| `MAX_RESPONSE_LENGTH` | Lunghezza massima della risposta generata            |
| `EVAL_PROMPT`       | Prompt usato per la preview dopo il training          |

## Struttura dei file principali

| File               | Descrizione breve                                                          |
|--------------------|---------------------------------------------------------------------------|
| `tuning.py`        | Centro di controllo dei parametri del modello (es. window size, numero neuroni, learning rate, batch size, ecc.) |
| `simple_tokenizer.py` | Vocabolario statico, dizionari di conversione carattere ↔ indice e funzione per generare sequenze one-hot |
| `neuron.py`        | Implementazione della classe `Neuron` con attivazione Leaky ReLU, forward e backpropagation |
| `layer.py`         | Implementazione di un layer denso composto da più neuroni e relative funzioni di forward e backward |
| `rnn_cell.py`      | Implementazione della cella RNN base, con pesi, bias, forward e gestione stato nascosto |
| `rnn_layer.py`     | Layer ricorrente che utilizza la `RNNCell` per processare sequenze temporali e gestire il training |
| `network.py`       | Costruzione e gestione dell’intera rete neurale (RNN + layer densi), con funzioni di forward e setup |
| `utils.py`         | Funzioni di utilità come softmax, encoding/decoding, cross-entropy loss, generazione testo e validazione vocabolario |
| `training.py`      | Loop di training del modello, gestione batching, calcolo loss e backpropagation su tutti i layer |
| `persistence.py`   | Funzioni di salvataggio e caricamento del modello completo in formato JSON |
| `main.py`       | Programma principale che avvia il chatbot da terminale, gestisce input utente, comandi e training incrementale |
| `log.txt`          | File di testo contenente Il contenuto definito dall' utente per il deep-train. Qualsiasi cosa incollata qui dentro allenerá il modello. |

---

## Disclaimer

Il codice è fornito **as-is** e potrebbe contenere bug, errori strutturali o limiti funzionali.  
Questo progetto è un **work in progress**: il funzionamento e la completezza del sistema non sono pertanto garantiti.  

I commenti presenti all’interno degli script sono stati generati con ChatGPT 4.1-mini.

Il progetto può essere utilizzato, distribuito e forkato liberamente senza alcuna autorizzazione da parte mia.  
Se trovate errori, bug o avete suggerimenti, potete segnalarli all’indirizzo email:  
`francesco.peruzzi.developer@gmail.com`

--Last Update: 30/06/2025--