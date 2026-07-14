from dataclasses import dataclass

from app.rag.ingest import get_embedder, get_store


@dataclass
class RetrievedChunk:
    text: str
    source: str
    unit: str
    distance: float
    kind: str  # "exact" | "file" | "semantic" — ver build_context/sources en llm.py


def embed_query(query: str) -> list[float]:
    return get_embedder().encode([f"query: {query}"], normalize_embeddings=True).tolist()[0]


def retrieve(query: str, top_k: int = 8) -> list[RetrievedChunk]:
    store = get_store()
    if store.count() == 0:
        return []

    # Orden de prioridad: 1) coincidencia exacta de un identificador (usuario,
    # PIN, IP...); 2) el archivo que la pregunta nombra explicitamente (ej.
    # "directorio JP"), salvo que sea el MISMO archivo del punto 1 (ahi ya
    # tenemos el registro preciso, una muestra generica de filas solo mete
    # ruido); 3) similitud semantica. Se evitan duplicados.
    exact = store.keyword_search(query, limit=3)
    exact_sources = {r["metadata"]["source"] for r in exact}

    file_matches: list[dict] = []
    for source in store.match_sources(query, max_results=2):
        if source in exact_sources:
            continue
        file_matches += store.get_by_source(source, limit=6)

    semantic = store.query(embed_query(query), top_k=top_k)

    tagged = (
        [("exact", r) for r in exact]
        + [("file", r) for r in file_matches]
        + [("semantic", r) for r in semantic]
    )

    seen: set[tuple[str, str]] = set()
    results: list[tuple[str, dict]] = []
    for kind, r in tagged:
        key = (r["metadata"]["source"], r["document"])
        if key in seen:
            continue
        seen.add(key)
        results.append((kind, r))

    limit = top_k + len(exact) + len(file_matches)
    return [
        RetrievedChunk(
            text=r["document"],
            source=r["metadata"]["source"],
            unit=r["metadata"]["unit"],
            distance=r["distance"],
            kind=kind,
        )
        for kind, r in results[:limit]
    ]


def list_sources() -> dict[str, int]:
    return get_store().list_sources()


def list_template_units() -> list[tuple[str, str]]:
    return get_store().list_template_units()


def get_template_text(source: str, unit: str) -> str | None:
    return get_store().get_template_text(source, unit)


def retrieve_template(query: str) -> RetrievedChunk | None:
    """Busca la plantilla mas parecida a la pregunta, comparando solo entre
    plantillas (no contra todo el corpus, ver query_templates). None si ni la
    mejor coincidencia se parece lo suficiente (evita devolver cualquier cosa
    para una pregunta que no es realmente sobre una plantilla)."""
    store = get_store()
    matches = store.query_templates(embed_query(query), top_k=1)
    if not matches or matches[0]["distance"] > 0.35:
        return None
    r = matches[0]
    return RetrievedChunk(
        text=r["document"],
        source=r["metadata"]["source"],
        unit=r["metadata"]["unit"],
        distance=r["distance"],
        kind="template",
    )
