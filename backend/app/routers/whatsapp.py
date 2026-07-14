import hashlib
import hmac
import json
from collections import deque

import httpx
from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.config import load_allowed_numbers, normalize_number, settings
from app.rag.llm import answer_question

router = APIRouter(prefix="/webhooks", tags=["whatsapp"])

MAX_WHATSAPP_TEXT = 4096

# OpenWA entrega "al menos una vez": un timeout o error de red puede hacer que
# reintente un webhook ya procesado. Se recuerdan las ultimas idempotencyKey
# vistas para no responder el mismo mensaje dos veces.
_seen_keys: set[str] = set()
_seen_keys_order: deque[str] = deque(maxlen=500)


def _already_processed(key: str) -> bool:
    if key in _seen_keys:
        return True
    if len(_seen_keys_order) == _seen_keys_order.maxlen:
        _seen_keys.discard(_seen_keys_order.popleft())
    _seen_keys_order.append(key)
    _seen_keys.add(key)
    return False


def _verify_signature(raw_body: bytes, signature_header: str | None) -> bool:
    if not settings.openwa_webhook_secret:
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(settings.openwa_webhook_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature_header[len("sha256=") :], expected)


def send_text(chat_id: str, text: str) -> None:
    url = f"{settings.openwa_base_url}/api/sessions/{settings.openwa_session_id}/messages/send-text"
    headers = {"X-API-Key": settings.openwa_api_key}
    text = text[: MAX_WHATSAPP_TEXT - 3] + "..." if len(text) > MAX_WHATSAPP_TEXT else text
    try:
        response = httpx.post(url, json={"chatId": chat_id, "text": text}, headers=headers, timeout=30)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        # Sin esto, si la sesion de WhatsApp se cae o la API key vence, el bot
        # deja de responderle a todo el mundo sin dejar ningun rastro.
        print(f"[whatsapp] error enviando mensaje a {chat_id}: {exc}")


def _resolve_lid_phone(contact_id: str) -> str | None:
    url = f"{settings.openwa_base_url}/api/sessions/{settings.openwa_session_id}/contacts/{contact_id}/phone"
    headers = {"X-API-Key": settings.openwa_api_key}
    try:
        response = httpx.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        # ValueError cubre json invalido (ej. pagina de error de un proxy):
        # response.json() no es un httpx.HTTPError y antes quedaba sin atajar,
        # tirando abajo el webhook completo para cualquier remitente "@lid".
        return None
    return data.get("phone") if isinstance(data, dict) else None


def _log_sender(chat_id: str) -> None:
    if chat_id.endswith("@lid"):
        phone = _resolve_lid_phone(chat_id)
        if phone:
            print(f"[whatsapp] LID {chat_id} -> resuelto a numero {phone}")
        else:
            print(f"[whatsapp] LID {chat_id} -> WhatsApp todavia no expone el numero real")
    else:
        print(f"[whatsapp] numero {chat_id}")


@router.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    raw_body = await request.body()
    if not _verify_signature(raw_body, request.headers.get("X-OpenWA-Signature")):
        raise HTTPException(status_code=401, detail="Firma de webhook invalida")

    payload = json.loads(raw_body)
    if payload.get("event") != "message.received":
        return {"ignored": True}

    idempotency_key = payload.get("idempotencyKey")
    if idempotency_key and _already_processed(idempotency_key):
        return {"ignored": True, "reason": "duplicado"}

    data = payload.get("data", {})
    if data.get("isGroup"):
        return {"ignored": True}

    chat_id = data.get("from") or ""
    body = (data.get("body") or "").strip()
    if not chat_id or not body:
        return {"ignored": True}

    # Para remitentes "@lid" (identificador interno de WhatsApp) el numero real
    # viene aparte en "senderPhone"; "from" no es un numero de telefono en ese
    # caso. Para todo lo demas se valida el propio chat_id: es el mismo
    # identificador al que despues se le manda la respuesta, asi no puede haber
    # un mensaje que pase la whitelist validando un campo pero conteste a otro.
    sender_for_whitelist = data.get("senderPhone") or chat_id if chat_id.endswith("@lid") else chat_id

    allowed_numbers = load_allowed_numbers()
    normalized = normalize_number(sender_for_whitelist)
    if normalized not in allowed_numbers:
        return {"ignored": True, "reason": "numero no autorizado"}

    # Se loguea/resuelve el LID solo despues de confirmar whitelist: no tiene
    # sentido hacerle una llamada HTTP a OpenWA por un mensaje de alguien no
    # autorizado, y evita que un fallo en esa llamada (ver _resolve_lid_phone)
    # afecte el procesamiento de mensajes de gente no autorizada.
    _log_sender(chat_id)

    # answer_question (embeddings + llamadas a Gemini/Groq) y send_text (HTTP a
    # OpenWA) son sincronicas y bloqueantes; llamarlas directo en este handler
    # async trabaria el event loop entero mientras duran, dejando a cualquier
    # otro mensaje/health-check esperando en fila. run_in_threadpool las saca
    # del loop principal.
    result = await run_in_threadpool(answer_question, body, chat_id=chat_id)
    reply = result.text
    if result.sources:
        reply += "\n\nFuentes: " + ", ".join(result.sources)

    await run_in_threadpool(send_text, chat_id, reply)
    return {"ok": True}
