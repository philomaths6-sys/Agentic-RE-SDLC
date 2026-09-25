# rag/retriever.py
"""Simple retrieval wrapper for Qdrant.

Usage::
    from rag.retriever import retrieve
    results = retrieve(query, top_k=5)

The function returns a list of payload dictionaries with the most similar
documents based on cosine similarity.
"""

import os
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

# Configuration – keep in sync with ``ingest.py``
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)

_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
_model = SentenceTransformer(EMBEDDING_MODEL)


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """Return the *top_k* most similar documents for *query*.

    The returned list contains the raw payloads stored in Qdrant, each
    payload includes the original ``text`` and any metadata (e.g. ``source``).
    """
    # Encode the query (CPU‑only, same model used for ingestion)
    query_vec = _model.encode([query])[0].tolist()

    # Search in Qdrant
    hits = _client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        score_threshold=None,
    )

    # Extract payloads (payload is a dict we stored during ingestion)
    results = []
    for hit in hits:
        payload = hit.payload or {}
        payload.setdefault("score", hit.score)
        payload.setdefault("citation", payload.get("filename", payload.get("source", "Regulatory KB")))
        results.append(payload)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m rag.retriever <query>")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    for r in retrieve(q):
        print(r.get("source", "<unknown>"))
        print(r.get("text", ""))
        print("---")
