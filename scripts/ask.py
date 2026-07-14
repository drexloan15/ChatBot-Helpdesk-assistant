"""CLI para probar el asistente por terminal, sin depender de WhatsApp.

Uso: python scripts/ask.py "cual es el procedimiento para..."
     python scripts/ask.py --list        (lista todo lo indexado, sin llamar a la IA)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.rag.ingest import get_store  # noqa: E402
from app.rag.llm import answer_question  # noqa: E402


def list_sources() -> None:
    store = get_store()
    sources = store.list_sources()
    if not sources:
        print("No hay documentos indexados todavia.")
        return
    print(f"{len(sources)} archivos indexados ({store.count()} chunks en total):\n")
    for source, n in sources.items():
        print(f"  {source} -> {n} chunks")


def main() -> None:
    if len(sys.argv) < 2:
        print('Uso: python scripts/ask.py "tu pregunta"  |  python scripts/ask.py --list')
        sys.exit(1)

    if sys.argv[1] == "--list":
        list_sources()
        return

    query = " ".join(sys.argv[1:])
    result = answer_question(query)

    print(f"\n{result.text}\n")
    if result.sources:
        print("Fuentes:", ", ".join(result.sources))


if __name__ == "__main__":
    main()
