# rag/retriever.py
"""Advanced Hybrid RAG Retriever for Sovereign Indian Financial Regulations.

Features:
1. Dual Collection Retrieval:
   - Queries 'statutory_corpus' by default, with optional user document augmentation from 'user_documents'.
2. Hybrid Search (Dense Cosine Vectors + Sparse BM25-style Lexical Scoring):
   - Fuses dense semantic similarity with exact regulatory token matching via Reciprocal Rank Fusion (RRF).
3. Minimum Similarity Threshold Enforcement:
   - Default threshold 0.65 as specified in statutory engineering standards, with adaptive fallback.
4. Reranker & Active Circular Prioritization:
   - Promotes ACTIVE circulars and deprioritizes SUPERSEDED circulars.
5. Structured Authoritative Citations:
   - Returns structured metadata (issuer, circular_no, circular_date, clause_no, status, score).
"""

import os
import re
import math
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
STATUTORY_COLLECTION = os.getenv("STATUTORY_COLLECTION", "statutory_corpus")
USER_DOCS_COLLECTION = os.getenv("USER_DOCS_COLLECTION", "user_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.65"))

_client: Optional[QdrantClient] = None
_model: Optional[SentenceTransformer] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    return _client


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ---------------------------------------------------------------------------
# BM25 & Lexical Scoring Helpers
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    """Extract lowercased alphanumeric tokens from text."""
    return re.findall(r"\b[a-zA-Z0-9_\-\./]+\b", text.lower())


def _compute_bm25_score(query_tokens: List[str], doc_tokens: List[str], avg_doc_len: float = 120.0, k1: float = 1.5, b: float = 0.75) -> float:
    """Compute BM25-style lexical relevance score between query and document."""
    if not doc_tokens:
        return 0.0
    doc_len = len(doc_tokens)
    score = 0.0
    doc_freq = {}
    for t in doc_tokens:
        doc_freq[t] = doc_freq.get(t, 0) + 1

    for qt in query_tokens:
        if qt in doc_freq:
            tf = doc_freq[qt]
            # Standard BM25 term frequency saturation
            idf = 1.0  # Normalized term weight
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * (numerator / denominator)
    return score


def _reciprocal_rank_fusion(dense_hits: List[Dict[str, Any]], query: str, k: int = 60) -> List[Dict[str, Any]]:
    """Rerank candidates using Hybrid Reciprocal Rank Fusion (Dense + Lexical)."""
    if not dense_hits:
        return []

    q_tokens = _tokenize(query)
    # Regulatory identifiers that deserve high lexical weight
    statutory_tokens = {"rbi", "sebi", "npci", "uidai", "dpdp", "pmla", "fiu", "fips", "iso", "cscrf", "autopay", "aadhaar", "hsm", "cbs", "fix"}
    has_statutory_key = set(q_tokens).intersection(statutory_tokens)

    # Score each hit
    scored_items = []
    for rank_idx, hit in enumerate(dense_hits):
        doc_text = hit.get("text", "")
        doc_tokens = _tokenize(doc_text)
        bm25 = _compute_bm25_score(q_tokens, doc_tokens)

        # Base dense rank
        dense_rank_score = 1.0 / (k + rank_idx + 1)
        bm25_rank_score = bm25 / (10.0 + bm25) if bm25 > 0 else 0.0

        # Status adjustment (penalize superseded circulars)
        status = hit.get("status", "ACTIVE").upper()
        status_multiplier = 1.0 if status == "ACTIVE" else 0.70

        # Exact statutory keyword boost
        stat_boost = 0.15 if any(st in doc_text.lower() for st in has_statutory_key) else 0.0

        fused_score = (dense_rank_score * 0.6 + bm25_rank_score * 0.4 + stat_boost) * status_multiplier
        scored_items.append((fused_score, hit))

    # Sort descending by fused score
    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_items]


# ---------------------------------------------------------------------------
# Primary Retrieval API
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    top_k: int = 5,
    score_threshold: Optional[float] = DEFAULT_SCORE_THRESHOLD,
    include_user_docs: bool = True,
    session_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Execute Hybrid Dense + Lexical retrieval across Qdrant statutory & user corpus.
    
    Args:
        query: User search query or software requirement text.
        top_k: Maximum number of ranked clauses to return.
        score_threshold: Minimum cosine similarity threshold (default 0.65).
        include_user_docs: If True, also searches user_documents collection.
        session_id: Optional session identifier to filter user documents.
        
    Returns:
        List of clause dictionaries with rich statutory metadata and citations.
    """
    if not query or not query.strip():
        return []

    client = get_client()
    model = get_model()

    try:
        query_vec = model.encode([query])[0].tolist()
    except Exception as exc:
        print(f"[WARN] Embedding generation failed: {exc}")
        return []

    # Determine collections to search
    target_collections = []
    # Primary statutory collection
    for col_candidate in [STATUTORY_COLLECTION, "documents", "finance_compliance_kb"]:
        try:
            if client.collection_exists(col_candidate):
                target_collections.append(col_candidate)
                break
        except Exception:
            pass

    if include_user_docs:
        try:
            if client.collection_exists(USER_DOCS_COLLECTION):
                target_collections.append(USER_DOCS_COLLECTION)
        except Exception:
            pass

    if not target_collections:
        return []

    raw_hits = []
    # Fetch candidates from each collection
    for col in target_collections:
        try:
            # Query Qdrant with candidate over-sampling for hybrid reranking
            search_limit = max(top_k * 3, 15)
            hits = client.search(
                collection_name=col,
                query_vector=query_vec,
                limit=search_limit,
                score_threshold=score_threshold
            )
            # If strict threshold returned zero hits, fallback with threshold=0.50
            if not hits and score_threshold is not None and score_threshold > 0.50:
                hits = client.search(
                    collection_name=col,
                    query_vector=query_vec,
                    limit=search_limit,
                    score_threshold=0.50
                )

            for h in hits:
                payload = dict(h.payload or {})
                payload["score"] = float(h.score)
                payload["point_id"] = str(h.id)
                raw_hits.append(payload)
        except Exception as search_err:
            print(f"[WARN] Qdrant search in collection '{col}' error: {search_err}")

    if not raw_hits:
        return []

    # Apply Hybrid BM25 + Dense RRF Reranking
    reranked = _reciprocal_rank_fusion(raw_hits, query)

    # Format structured citations and deduplicate by text hash
    seen_texts = set()
    final_results = []
    for r in reranked:
        txt = r.get("text", "").strip()
        t_hash = hash(txt[:100])
        if t_hash in seen_texts:
            continue
        seen_texts.add(t_hash)

        # Standardize citation format
        issuer = r.get("issuer", "Regulatory Authority")
        title = r.get("doc_title", r.get("filename", "Statutory Document"))
        date = r.get("circular_date", "2024")
        clause = r.get("clause_no", "Clause Ref")
        status = r.get("status", "ACTIVE")

        authoritative_citation = r.get("citation") or f"{issuer} {title} ({date}), {clause} [{status}]"
        r["citation"] = authoritative_citation
        final_results.append(r)
        if len(final_results) >= top_k:
            break

    return final_results


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Aadhaar Data Vault HSM isolation requirements"
    print(f"Testing retrieval for: '{q}'")
    results = retrieve(q, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] {res['citation']} (Score: {res['score']:.4f})")
        print(f"    Excerpt: {res['text'][:150]}...")
