from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import os

# Carica il token da secrets.txt
def load_token(filepath="secrets.txt"):
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("BOT_TOKEN="):
                return line.strip().split("=", 1)[1]
    raise Exception("BOT_TOKEN non trovato in secrets.txt")

# Funzione che risponde ai messaggi
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user.first_name or "utente"
    message = update.message.text
    print(f"[Messaggio da {user}]: {message}")
    await update.message.reply_text(f"Ciao {user}! Hai detto: {message}")

# Esegui il bot
def main():
    token = load_token()
    app = ApplicationBuilder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Bot avviato. In ascolto...")
    app.run_polling()

if __name__ == "__main__":
    main()