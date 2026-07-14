from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.rag.llm import answer_question


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    # Fail-closed: sin INTERNAL_API_KEY configurada el endpoint queda
    # inutilizable en vez de abierto a cualquiera que llegue al puerto del
    # backend (este endpoint no pasa por la whitelist de WhatsApp).
    if not settings.internal_api_key or x_api_key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="API key invalida o no configurada")


router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(_require_api_key)])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = answer_question(request.message)
    return ChatResponse(answer=result.text, sources=result.sources)
