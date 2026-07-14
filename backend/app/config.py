import json
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    # Mucha mas cuota diaria que gemini_model, se usa como segundo escalon
    # antes de caer a Groq (ver app/rag/llm.py:get_providers).
    gemini_model_fallback: str = "gemini-3.1-flash-lite"

    embedding_model: str = "intfloat/multilingual-e5-small"

    vectorstore_path: str = "./vectorstore"

    docs_path: str = "../docs"

    openwa_base_url: str = "http://localhost:2785"
    openwa_api_key: str = ""
    openwa_session_id: str = ""
    openwa_webhook_secret: str = ""

    # Protege POST /chat (sin esto, cualquiera que llegue al puerto del backend
    # puede consultar todo el RAG sin pasar por la whitelist de WhatsApp).
    internal_api_key: str = ""

    @field_validator("vectorstore_path", "docs_path")
    @classmethod
    def _resolve_relative_to_backend_dir(cls, v: str) -> str:
        # Las rutas relativas del .env deben ser independientes del directorio
        # desde el que se ejecute el script (backend/, raiz del repo, etc.).
        path = Path(v)
        if not path.is_absolute():
            path = (BACKEND_DIR / path).resolve()
        return str(path)


settings = Settings()


def load_allowed_numbers() -> set[str]:
    path = BACKEND_DIR / "allowed_numbers.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Archivo ausente o mal editado: no autorizar a nadie (fail-closed) en
        # vez de tirar un 500 en cada mensaje entrante hasta que se note.
        return set()
    return {normalize_number(n) for n in data.get("allowed_numbers", [])}


def normalize_number(raw: str) -> str:
    """Deja solo digitos de un identificador de WhatsApp (numero o '...@lid'/'...@c.us';
    las letras del sufijo se descartan solas al filtrar por isdigit)."""
    return "".join(ch for ch in raw if ch.isdigit())
