"""Vector store minimalista basado en numpy (sin dependencias que requieran compilar).

Guarda los embeddings en un .npy y los documentos/metadata en un .json,
ambos en el mismo directorio. Alcanza de sobra para el volumen de
manuales de una empresa (miles de chunks) con busqueda por fuerza bruta.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Direcciones IP como un solo token (si no, "10.0.16.240" se parte en "10",
# "0", "16", "240" y se pierde como identificador). Se prueba primero para
# que gane sobre el patron de palabra generica.
_WORD_RE = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|[\wÀ-ÿ]{3,}")

# Nombre de unidad tipo "TICKET_COLA_PRHP": todo mayusculas y guion bajo, como
# lo genera _split_ticket_sections en ingest.py (distinto de los nombres
# genericos "pagina 1", "hoja 'X'", "documento completo", etc.)
_TAG_UNIT_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Interrogativos/palabras funcionales: nunca sirven como identificador aunque
# por casualidad aparezcan en pocos chunks del corpus (ej. "cual" solo sale en
# un puñado de manuales en prosa, asi que su frecuencia da "rara" sin serlo -
# eso hacia que keyword_search devolviera un PDF sin relacion como "coincidencia
# exacta" con cualquier pregunta que empezara con "cual es...").
_NON_IDENTIFYING_WORDS = {
    "que", "hay", "cual", "cuales", "como", "donde", "cuando", "cuanto",
    "cuantos", "cuantas", "quien", "quienes", "esta", "estan", "son",
    "tiene", "tienen", "este", "estos", "esas", "esos", "esa", "ese",
    "para", "con", "sin", "los", "las", "del", "dame", "dime",
}


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _shares_stem(a: str, b: str, min_len: int = 5, min_ratio: float = 0.75) -> bool:
    """Compara por prefijo comun para tolerar variaciones de una misma raiz
    (derivar/derivacion/derivado, configurar/configuracion...). Exige que el
    prefijo compartido sea una proporcion alta de la palabra mas corta, no
    solo una cantidad fija de letras — si no, palabras sin relacion que
    arrancan igual (servicio/servidor) se confunden entre si."""
    if a == b:
        return True
    n = min(len(a), len(b))
    if n < min_len:
        return False
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= min_len and common / n >= min_ratio


class SimpleVectorStore:
    def __init__(self, path: str) -> None:
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._embeddings_file = self.dir / "embeddings.npy"
        self._records_file = self.dir / "records.json"

        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._token_doc_freq: Counter | None = None
        self._load()

    def _load(self) -> None:
        if self._records_file.exists() and self._embeddings_file.exists():
            records = json.loads(self._records_file.read_text(encoding="utf-8"))
            self._ids = [r["id"] for r in records]
            self._documents = [r["document"] for r in records]
            self._metadatas = [r["metadata"] for r in records]
            self._embeddings = np.load(self._embeddings_file)

    def _save(self) -> None:
        # Escritura atomica (escribir a un temporal + rename): si el proceso
        # se cae a mitad de camino, records.json/embeddings.npy quedan
        # intactos en vez de truncados/desincronizados entre si.
        records = [
            {"id": id_, "document": doc, "metadata": meta}
            for id_, doc, meta in zip(self._ids, self._documents, self._metadatas)
        ]
        records_tmp = self.dir / "records.json.tmp"
        records_tmp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        records_tmp.replace(self._records_file)

        if self._embeddings is not None:
            embeddings_tmp = self.dir / "embeddings.tmp.npy"
            np.save(embeddings_tmp, self._embeddings)
            embeddings_tmp.replace(self._embeddings_file)

    def count(self) -> int:
        return len(self._ids)

    def list_sources(self) -> dict[str, int]:
        """Devuelve {archivo: cantidad de chunks} para todo lo indexado."""
        counts: dict[str, int] = {}
        for meta in self._metadatas:
            source = meta["source"]
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    def list_template_units(self) -> list[tuple[str, str]]:
        """Devuelve [(archivo, nombre_seccion)] para las plantillas indexadas
        (secciones etiquetadas '### TAG', ver _split_ticket_sections en
        ingest.py), sin duplicar si una seccion larga quedo partida en varios
        chunks. Sirve para responder 'que plantillas hay' con la lista
        completa en vez de depender de la busqueda semantica (que con top_k
        limitado solo trae unas pocas)."""
        seen: set[tuple[str, str]] = set()
        result: list[tuple[str, str]] = []
        for meta in self._metadatas:
            unit = meta["unit"]
            if not _TAG_UNIT_RE.match(unit):
                continue
            key = (meta["source"], unit)
            if key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    def get_template_text(self, source: str, unit: str) -> str | None:
        """Texto completo de una plantilla puntual (por si quedo partida en
        varios chunks, aunque en la practica las secciones de tickets.txt
        entran en uno solo)."""
        parts = [
            doc
            for doc, meta in zip(self._documents, self._metadatas)
            if meta["source"] == source and meta["unit"] == unit
        ]
        return "\n".join(parts) if parts else None

    def delete_source(self, source: str) -> int:
        """Borra todos los chunks de un archivo (para reemplazarlo sin dejar
        filas viejas mezcladas si el archivo nuevo tiene menos filas/paginas)."""
        keep = [i for i, meta in enumerate(self._metadatas) if meta["source"] != source]
        removed = len(self._ids) - len(keep)
        if removed:
            self._ids = [self._ids[i] for i in keep]
            self._documents = [self._documents[i] for i in keep]
            self._metadatas = [self._metadatas[i] for i in keep]
            self._embeddings = self._embeddings[keep] if self._embeddings is not None else None
            self._token_doc_freq = None
            self._save()
        return removed

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        new_embeddings = np.array(embeddings, dtype=np.float32)
        index_by_id = {id_: idx for idx, id_ in enumerate(self._ids)}

        for i, id_ in enumerate(ids):
            if id_ in index_by_id:
                idx = index_by_id[id_]
                self._documents[idx] = documents[i]
                self._metadatas[idx] = metadatas[i]
                self._embeddings[idx] = new_embeddings[i]
            else:
                self._ids.append(id_)
                self._documents.append(documents[i])
                self._metadatas.append(metadatas[i])
                if self._embeddings is None:
                    self._embeddings = new_embeddings[i : i + 1]
                else:
                    self._embeddings = np.vstack([self._embeddings, new_embeddings[i]])

        self._token_doc_freq = None  # el indice de rareza queda desactualizado
        self._save()

    def match_sources(self, query: str, max_results: int = 2, max_content_freq: int = 20) -> list[str]:
        """Encuentra archivos cuyo NOMBRE coincide con la pregunta (ej. 'directorio JP'
        -> 'Directorio JP al 04.05.2026.xlsx', o 'vpn' -> el pdf de VPN), aunque esas
        palabras no aparezcan dentro del contenido. Prioriza palabras poco comunes
        entre los nombres de archivo (ej. 'vpn') sobre las genericas (ej. 'manual',
        'caja arequipa') que se repiten en casi todos los titulos.

        Con pocos archivos totales, una palabra generica del rubro (ej. 'servidor')
        puede aparecer en solo 1-2 nombres de archivo por pura casualidad y parecer
        'rara' — por eso ademas se exige que sea poco frecuente en el CONTENIDO
        (pocos chunks la mencionan), que es la señal real de que es especifica."""
        query_words = _words(query)
        if not query_words:
            return []

        self._ensure_token_doc_freq()

        all_sources = sorted({meta["source"] for meta in self._metadatas})
        name_words_by_source = {s: _words(Path(s).stem) for s in all_sources}

        file_freq: Counter = Counter()
        for name_words in name_words_by_source.values():
            file_freq.update(name_words)

        scored = []
        for source, name_words in name_words_by_source.items():
            score = 0.0
            has_distinctive_match = False
            for name_word in name_words:
                matched = name_word in query_words or any(
                    _shares_stem(qw, name_word) for qw in query_words
                )
                if not matched:
                    continue
                rarity = 1.0 / file_freq[name_word]
                score += rarity
                if file_freq[name_word] <= 2 and self._token_doc_freq.get(name_word, 0) <= max_content_freq:
                    has_distinctive_match = True
            if has_distinctive_match:
                scored.append((score, source))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [source for _, source in scored[:max_results]]

    def get_by_source(self, source: str, limit: int | None = None) -> list[dict]:
        matches = [
            {"document": doc, "metadata": meta, "distance": 0.0}
            for doc, meta in zip(self._documents, self._metadatas)
            if meta["source"] == source
        ]
        return matches[:limit] if limit else matches

    def _ensure_token_doc_freq(self) -> None:
        if self._token_doc_freq is not None:
            return
        freq: Counter = Counter()
        for doc in self._documents:
            freq.update(_words(doc))
        self._token_doc_freq = freq

    def keyword_search(self, query: str, limit: int = 3, max_doc_freq: int = 15) -> list[dict]:
        """Busca coincidencia exacta de tokens 'identificadores' (usuario, IP,
        serie, PIN, etc.): palabras de la pregunta que aparecen en MUY pocos
        chunks del corpus (no palabras genericas como 'manual' o 'sistema',
        que aparecen en decenas de documentos y no sirven para desambiguar).
        Si el chunk tiene varias filas (datos tabulares), devuelve solo la(s)
        fila(s) que realmente coinciden, para que la IA no confunda una fila
        con la de al lado (ej. dos usuarios con nombres muy parecidos)."""
        self._ensure_token_doc_freq()
        candidates = _words(query) - _NON_IDENTIFYING_WORDS
        tokens = set(t for t in candidates if 0 < self._token_doc_freq.get(t, 0) <= max_doc_freq)
        if not tokens:
            return []

        matches = []
        for doc, meta in zip(self._documents, self._metadatas):
            if not (_words(doc) & tokens):
                continue

            snippet = doc
            if meta.get("row_based"):
                # Solo tiene sentido recortar a la fila que matchea en datos
                # tabulares (Excel/CSV); en prosa (manuales, plantillas de
                # tickets) el chunk completo es la unidad de sentido y cortar
                # por linea perderia contexto.
                lines = [ln for ln in doc.split("\n") if ln.strip()]
                matching_lines = [ln for ln in lines if _words(ln) & tokens]
                snippet = "\n".join(matching_lines) if matching_lines else doc

            matches.append({"document": snippet, "metadata": meta, "distance": 0.0})
            if len(matches) >= limit:
                break
        return matches

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if not self._ids:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        similarities = self._embeddings @ q  # embeddings normalizados -> coseno
        top_k = min(top_k, len(self._ids))
        top_idx = np.argpartition(-similarities, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-similarities[top_idx])]

        return [
            {
                "document": self._documents[i],
                "metadata": self._metadatas[i],
                "distance": 1 - float(similarities[i]),
            }
            for i in top_idx
        ]

    def query_templates(self, query_embedding: list[float], top_k: int = 1) -> list[dict]:
        """Como query(), pero comparando solo contra las plantillas (secciones
        '### TAG', ver list_template_units). Si se compara contra TODO el
        corpus, una pregunta como 'plantilla de solicitud de pin' pierde
        contra los miles de filas de la planilla de PINs, que mencionan 'pin'
        muchas mas veces que las dos plantillas de tickets sobre PIN."""
        indices = [i for i, meta in enumerate(self._metadatas) if _TAG_UNIT_RE.match(meta["unit"])]
        if not indices:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        sub_embeddings = self._embeddings[indices]
        similarities = sub_embeddings @ q
        top_k = min(top_k, len(indices))
        top_idx = np.argpartition(-similarities, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-similarities[top_idx])]

        return [
            {
                "document": self._documents[indices[i]],
                "metadata": self._metadatas[indices[i]],
                "distance": 1 - float(similarities[i]),
            }
            for i in top_idx
        ]
