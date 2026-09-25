# rag/ingest.py
"""Ingestion script for loading documents into Qdrant.

This module scans the data/ folder for text, markdown, docx, and pdf files,
creates (or recreates) a Qdrant collection called 'documents' and inserts
embeddings for each document.
"""

import os
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION', 'documents')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')


def _read_file_content(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in ['.txt', '.md']:
        return path.read_text(encoding='utf-8', errors='ignore')
    elif ext == '.docx':
        try:
            import docx
            doc = docx.Document(path)
            return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception as exc:
            print(f'[WARN] Failed to parse DOCX {path}: {exc}')
            return ''
    elif ext == '.pdf':
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            text = ''
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + '\n'
            return text
        except Exception as exc:
            print(f'[WARN] Failed to parse PDF {path}: {exc}')
            return ''
    return ''


def _load_documents(root_dir: Path) -> List[dict]:
    """Recursively load documents (.txt, .md, .docx, .pdf) from root_dir."""
    docs = []
    doc_id = 0
    supported_extensions = {'.txt', '.md', '.docx', '.pdf'}
    
    for path in root_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            text = _read_file_content(path)
            if text.strip():
                docs.append({'id': doc_id, 'text': text, 'metadata': {'source': str(path), 'filename': path.name}})
                doc_id += 1
    return docs


def _chunk_text(text: str, chunk_size: int = 750, overlap: int = 75) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def main(data_dir: str = 'data') -> None:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    model = SentenceTransformer(EMBEDDING_MODEL)

    raw_docs = _load_documents(Path(data_dir))
    if not raw_docs:
        print('[INFO] No documents found to ingest.')
        return

    chunked_docs: List[dict] = []
    for doc in raw_docs:
        for chunk in _chunk_text(doc['text']):
            chunked_docs.append(
                {'id': len(chunked_docs), 'text': chunk, 'metadata': doc['metadata']}
            )

    if not chunked_docs:
        print('[INFO] No non-empty text chunks generated.')
        return

    sample_vector = model.encode([chunked_docs[0]['text']])[0]
    vector_size = len(sample_vector)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )

    ids = [doc['id'] for doc in chunked_docs]
    texts = [doc['text'] for doc in chunked_docs]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True).tolist()
    payloads = [
        {'text': txt, **doc['metadata']} for txt, doc in zip(texts, chunked_docs)
    ]

    # Batch upserts to prevent HTTP timeout on large payloads
    upsert_batch_size = 200
    total_points = len(ids)
    for i in range(0, total_points, upsert_batch_size):
        end_idx = min(i + upsert_batch_size, total_points)
        batch_points = [
            rest.PointStruct(id=_id, vector=vec, payload=payload)
            for _id, vec, payload in zip(ids[i:end_idx], vectors[i:end_idx], payloads[i:end_idx])
        ]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch_points,
            wait=True,
        )

    print(f'[INFO] Ingested {len(chunked_docs)} chunks from {len(raw_docs)} files into collection "{COLLECTION_NAME}".')


if __name__ == '__main__':
    main()
