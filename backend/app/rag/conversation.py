"""Memoria de conversacion simple, en memoria del proceso (no persiste a disco).

Guarda los ultimos turnos por remitente para que preguntas de seguimiento
("y su correo?", "y su nombre?") se puedan resolver contra el tema de la
pregunta anterior. Se pierde si se reinicia el backend - alcanza para el
volumen de un asistente interno por WhatsApp.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

MAX_TURNS = 4
TTL_SECONDS = 30 * 60  # 30 min de inactividad -> se olvida el tema


@dataclass
class _Conversation:
    turns: list[tuple[str, str]] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)


_conversations: dict[str, _Conversation] = {}


def get_history(chat_id: str) -> list[tuple[str, str]]:
    convo = _conversations.get(chat_id)
    if not convo:
        return []
    if time.time() - convo.last_seen > TTL_SECONDS:
        _conversations.pop(chat_id, None)
        return []
    return convo.turns


def append(chat_id: str, question: str, answer: str) -> None:
    convo = _conversations.setdefault(chat_id, _Conversation())
    convo.turns.append((question, answer))
    del convo.turns[:-MAX_TURNS]
    convo.last_seen = time.time()


def clear(chat_id: str) -> None:
    _conversations.pop(chat_id, None)
