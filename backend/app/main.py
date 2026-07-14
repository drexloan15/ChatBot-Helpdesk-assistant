from fastapi import FastAPI

from app.rag.ingest import get_store
from app.routers import chat, whatsapp

app = FastAPI(title="Asistente IA interno")

app.include_router(chat.router)
app.include_router(whatsapp.router)


@app.get("/health")
def health():
    # No llama a Gemini/Groq/OpenWA (gastaria cuota/latencia en cada check),
    # pero al menos expone si el vectorstore cargo datos: antes "ok" no decia
    # nada sobre si el RAG tenia algo para responder.
    return {"status": "ok", "vectorstore_chunks": get_store().count()}
