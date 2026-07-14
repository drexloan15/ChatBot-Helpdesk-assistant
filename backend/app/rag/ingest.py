"""Ingesta de manuales (PDF, Word, Excel, imagenes, texto) hacia Chroma.

Uso: python -m app.rag.ingest
Recorre DOCS_PATH, extrae texto de cada archivo soportado, lo trocea
y guarda los embeddings en la coleccion de Chroma configurada.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from docx import Document as DocxDocument
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.rag.vectorstore import SimpleVectorStore

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}
EXCEL_EXT = {".xlsx", ".xls"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
TEXT_EXT = {".txt", ".md"}

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def embed_passages(texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {t}" for t in texts]
    return get_embedder().encode(prefixed, normalize_embeddings=True).tolist()


_store: SimpleVectorStore | None = None


def get_store() -> SimpleVectorStore:
    global _store
    if _store is None:
        _store = SimpleVectorStore(settings.vectorstore_path)
    return _store


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = current[-overlap:] + "\n" + para if current else para
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap :]
    if current:
        chunks.append(current)
    return chunks


def extract_pdf(path: Path) -> list[tuple[str, str, bool]]:
    """Devuelve [(unidad, texto, row_based)] por pagina."""
    units = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                units.append((f"pagina {i}", text, False))
    return units


def extract_docx(path: Path) -> list[tuple[str, str, bool]]:
    doc = DocxDocument(path)
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [("documento completo", text, False)] if text.strip() else []


_DATE_LIKE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
_NUMERIC_RE = re.compile(r"^-?\d+([.,]\d+)?$")


def _looks_like_label(value: str) -> bool:
    """Un encabezado real es un texto corto tipo 'AGENCIA' o 'celular', no un
    numero, fecha, email o una nota larga (eso es lo que suele haber en las
    filas decorativas arriba del encabezado real)."""
    v = value.strip()
    if not v or len(v) > 40 or "@" in v:
        return False
    if _DATE_LIKE_RE.match(v) or _NUMERIC_RE.match(v):
        return False
    return True


def _find_header_row(path: Path, sheet_name, max_scan: int = 15) -> int:
    """Algunas planillas tienen filas decorativas (fecha, notas, totales) antes
    de la fila real de encabezados; si pandas toma esa fila como header, todas
    las columnas quedan como 'Unnamed: N' y se pierde el nombre del campo
    (ej. 'celular'). Se prueban las primeras filas y se elige la que mas
    'pinta de encabezado' tiene (celdas cortas tipo etiqueta, no datos)."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=str, nrows=max_scan)
    best_idx, best_score = 0, -1
    for idx in range(len(raw)):
        values = raw.iloc[idx].fillna("").astype(str).tolist()
        if sum(1 for v in values if v.strip()) < 3:
            continue  # fila casi vacia: probablemente decorativa, se ignora
        score = sum(1 for v in values if _looks_like_label(v))
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx


def extract_excel(path: Path) -> list[tuple[str, str, bool]]:
    units = []
    sheet_names = pd.ExcelFile(path).sheet_names
    for sheet_name in sheet_names:
        header_row = _find_header_row(path, sheet_name)
        df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, dtype=str)
        df = df.fillna("")
        rows = []
        for _, row in df.iterrows():
            row_text = " | ".join(
                f"{col}: {val}"
                for col, val in row.items()
                if str(val).strip() and not str(col).startswith("Unnamed:")
            )
            if row_text:
                rows.append(row_text)
        if rows:
            # "\n\n" para que cada fila sea su propio parrafo y chunk_text
            # nunca corte una fila a la mitad.
            units.append((f"hoja '{sheet_name}'", "\n\n".join(rows), True))
    return units


def extract_image(path: Path) -> list[tuple[str, str, bool]]:
    text = pytesseract.image_to_string(Image.open(path), lang="spa+eng").strip()
    return [("imagen (OCR)", text, False)] if text else []


def _detect_delimiter(text: str) -> str | None:
    first_line = text.splitlines()[0].strip() if text else ""
    marker = first_line.strip('"').lower()
    if marker.startswith("sep="):
        return marker[len("sep=") :][:1] or None

    sample = "\n".join(text.splitlines()[1:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        return None


def extract_delimited_text(text: str, delimiter: str) -> list[tuple[str, str, bool]]:
    lines = text.splitlines()
    if lines and lines[0].strip('"').lower().startswith("sep="):
        lines = lines[1:]

    rows = list(csv.reader(lines, delimiter=delimiter))
    if len(rows) < 2:
        return []

    header = rows[0]
    records = []
    for row in rows[1:]:
        row_text = " | ".join(f"{col}: {val}" for col, val in zip(header, row) if val.strip())
        if row_text:
            records.append(row_text)

    # "\n\n" para que cada fila sea su propio parrafo (ver extract_excel).
    return [("datos tabulares", "\n\n".join(records), True)] if records else []


_TICKET_TAG_RE = re.compile(r"^###\s+(\S+)\s*$")


def _split_ticket_sections(text: str) -> list[tuple[str, str]] | None:
    """Archivos de plantillas de tickets (ej. docs/tickets.txt): cada seccion
    arranca con una linea '### NOMBRE_UNICO'. Se usa para que cada plantilla
    quede como su propia unidad, sin importar el tamaño — nunca se fusiona con
    la de al lado (a diferencia del chunking generico por parrafos), y asi
    una pregunta sobre una plantilla no trae de arrastre el texto de otra."""
    lines = text.splitlines()
    tag_positions = [i for i, ln in enumerate(lines) if _TICKET_TAG_RE.match(ln.strip())]
    if not tag_positions:
        return None

    sections = []
    for idx, start in enumerate(tag_positions):
        tag = _TICKET_TAG_RE.match(lines[start].strip()).group(1)
        end = tag_positions[idx + 1] if idx + 1 < len(tag_positions) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        if body:
            sections.append((tag, body))
    return sections or None


def _read_text_auto_encoding(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
        return raw.decode("utf-8", errors="ignore")


def extract_plain_text(path: Path) -> list[tuple[str, str, bool]]:
    text = _read_text_auto_encoding(path).strip()
    if not text:
        return []

    ticket_sections = _split_ticket_sections(text)
    if ticket_sections:
        return [(tag, body, False) for tag, body in ticket_sections]

    delimiter = _detect_delimiter(text)
    if delimiter:
        units = extract_delimited_text(text, delimiter)
        if units:
            return units

    return [("documento completo", text, False)]


def extract_units(path: Path) -> list[tuple[str, str, bool]]:
    suffix = path.suffix.lower()
    if suffix in PDF_EXT:
        return extract_pdf(path)
    if suffix in DOCX_EXT:
        return extract_docx(path)
    if suffix in EXCEL_EXT:
        return extract_excel(path)
    if suffix in IMAGE_EXT:
        return extract_image(path)
    if suffix in TEXT_EXT:
        return extract_plain_text(path)
    return []


def iter_supported_files(docs_path: Path) -> Iterable[Path]:
    supported = PDF_EXT | DOCX_EXT | EXCEL_EXT | IMAGE_EXT | TEXT_EXT
    for path in docs_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported:
            yield path


def ingest_file(path: Path, docs_path: Path, store: SimpleVectorStore) -> int:
    units = extract_units(path)
    ids, docs, metadatas = [], [], []
    rel_path = str(path.relative_to(docs_path))
    for unit_name, unit_text, row_based in units:
        for i, chunk in enumerate(chunk_text(unit_text)):
            ids.append(f"{rel_path}::{unit_name}::{i}")
            docs.append(chunk)
            metadatas.append(
                {
                    "source": rel_path,
                    "unit": unit_name,
                    "type": path.suffix.lower(),
                    "row_based": row_based,
                }
            )

    # Se borra lo anterior de este archivo antes de recargar: si el archivo
    # nuevo tiene menos filas/paginas que el viejo, no quedan chunks huerfanos
    # de una version anterior mezclados con los datos actuales.
    store.delete_source(rel_path)

    if not docs:
        return 0

    embeddings = embed_passages(docs)
    store.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)
    return len(docs)


def main() -> None:
    docs_path = Path(settings.docs_path)
    if not docs_path.exists():
        print(f"No existe la carpeta de documentos: {docs_path}")
        sys.exit(1)

    store = get_store()
    total_files = 0
    total_chunks = 0
    for path in iter_supported_files(docs_path):
        n = ingest_file(path, docs_path, store)
        if n:
            total_files += 1
            total_chunks += n
            print(f"  {path.relative_to(docs_path)} -> {n} chunks")
        else:
            print(f"  {path.relative_to(docs_path)} -> sin texto extraible, omitido")

    print(f"\nListo: {total_files} archivos, {total_chunks} chunks en '{settings.vectorstore_path}'.")


if __name__ == "__main__":
    main()
