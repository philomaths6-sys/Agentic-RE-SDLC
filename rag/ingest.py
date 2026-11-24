# rag/ingest.py
"""Statutory & User Document Ingestion Layer for Qdrant.

Implements:
1. Dual Collection Architecture:
   - 'statutory_corpus': Authoritative, versioned repository of RBI, SEBI, NPCI, UIDAI, DPDP, PMLA mandates.
   - 'user_documents': Isolated storage for user-uploaded specifications and project drafts.
2. Incremental Upserts:
   - Uses deterministic hashing (SHA-256) per chunk to prevent duplicates and avoid wiping collections on upload.
3. Clause- & Section-Aware Chunking:
   - Integrates with rag.chunker to preserve legal clause numbering, issuers, dates, and active/superseded status.
4. Scheduled / On-Demand Versioning:
   - Tracks corpus version and document updates with audit metadata.
"""

import os
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

from rag.chunker import (
    extract_document_metadata,
    chunk_by_clause_or_section,
    REGULATORY_CATALOG
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
STATUTORY_COLLECTION = os.getenv("STATUTORY_COLLECTION", "statutory_corpus")
USER_DOCS_COLLECTION = os.getenv("USER_DOCS_COLLECTION", "user_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CORPUS_VERSION = os.getenv("STATUTORY_CORPUS_VERSION", "2026.11-v2")


def _deterministic_point_id(content_hash: str, index: int) -> str:
    """Generate deterministic UUIDv5 for a chunk to enable idempotent upserts."""
    seed = f"{content_hash}_{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def _read_file_content(path: Path) -> str:
    """Extract plain text from txt, md, docx, or pdf files."""
    ext = path.suffix.lower()
    if ext in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception as exc:
            print(f"[WARN] Failed to parse DOCX {path}: {exc}")
            return ""
    elif ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            text_parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
            return "\n".join(text_parts)
        except Exception as exc:
            print(f"[WARN] Failed to parse PDF {path}: {exc}")
            return ""
    return ""


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 384) -> None:
    """Ensure a Qdrant collection exists with cosine distance."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE)
        )
        print(f"[INFO] Created Qdrant collection '{collection_name}'.")


def ingest_statutory_corpus(
    data_dir: str = "data",
    force_recreate: bool = False
) -> Dict[str, Any]:
    """Ingest sovereign statutory circulars into 'statutory_corpus' with clause metadata.
    
    Preserves existing data and performs incremental upserts unless force_recreate=True.
    """
    client = get_qdrant_client()
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_sentence_embedding_dimension()

    if force_recreate and client.collection_exists(STATUTORY_COLLECTION):
        client.delete_collection(STATUTORY_COLLECTION)
        print(f"[INFO] Deleted collection '{STATUTORY_COLLECTION}' for clean rebuild.")

    ensure_collection(client, STATUTORY_COLLECTION, vector_size)

    # Establish aliases so legacy queries continue working
    for alias_name in ["documents", "finance_compliance_kb"]:
        try:
            client.update_collection_aliases(
                change_aliases_operations=[
                    rest.CreateAliasOperation(
                        create_alias=rest.CreateAlias(
                            collection_name=STATUTORY_COLLECTION,
                            alias_name=alias_name
                        )
                    )
                ]
            )
        except Exception:
            pass

    # Gather statutory knowledge base files
    kb_dirs = [
        Path(data_dir) / "knowledge_base",
        Path(data_dir) / "priority_reference",
        Path(data_dir) / "sample_srs"
    ]

    all_chunks: List[Dict[str, Any]] = []
    processed_files = 0

    for kb_dir in kb_dirs:
        if not kb_dir.exists():
            continue
        for file_path in kb_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in [".txt", ".md", ".docx", ".pdf"]:
                continue

            raw_text = _read_file_content(file_path)
            if not raw_text.strip():
                continue

            meta = extract_document_metadata(file_path)
            meta["corpus_version"] = CORPUS_VERSION
            meta["ingested_at"] = datetime.now(timezone.utc).isoformat()

            chunks = chunk_by_clause_or_section(raw_text, meta)
            file_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]

            for idx, ch in enumerate(chunks):
                pt_id = _deterministic_point_id(f"{meta['filename']}_{file_hash}", idx)
                ch["point_id"] = pt_id
                ch["chunk_index"] = idx
                all_chunks.append(ch)

            processed_files += 1

    if not all_chunks:
        print("[WARN] No statutory chunks generated.")
        return {"status": "empty", "chunks": 0}

    # Vectorize and batch upsert
    texts = [c["text"] for c in all_chunks]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    batch_size = 150
    total = len(all_chunks)
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        points = [
            rest.PointStruct(
                id=all_chunks[j]["point_id"],
                vector=vectors[j],
                payload=all_chunks[j]
            )
            for j in range(i, end_idx)
        ]
        client.upsert(collection_name=STATUTORY_COLLECTION, points=points, wait=True)

    print(f"[SUCCESS] Ingested {len(all_chunks)} clause-level chunks from {processed_files} statutory files into '{STATUTORY_COLLECTION}'.")
    return {
        "status": "success",
        "collection": STATUTORY_COLLECTION,
        "files_processed": processed_files,
        "chunks_ingested": len(all_chunks),
        "version": CORPUS_VERSION
    }


def ingest_user_document(
    file_path: Path,
    session_id: str = "default_session",
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Incrementally ingest a single user document into the isolated 'user_documents' collection.
    
    Does NOT touch or delete the statutory corpus!
    """
    client = get_qdrant_client()
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_sentence_embedding_dimension()

    ensure_collection(client, USER_DOCS_COLLECTION, vector_size)

    raw_text = _read_file_content(file_path)
    if not raw_text.strip():
        return {"status": "error", "message": "Document contains no readable text"}

    meta = {
        "source": str(file_path),
        "filename": file_path.name,
        "issuer": "User Document",
        "circular_no": f"USER-{file_path.stem}",
        "circular_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "doc_title": file_path.stem.replace("_", " "),
        "status": "ACTIVE",
        "version": "1.0",
        "domain": "user_spec",
        "is_statutory": False,
        "session_id": session_id,
        "user_id": user_id,
        "ingested_at": datetime.now(timezone.utc).isoformat()
    }

    chunks = chunk_by_clause_or_section(raw_text, meta)
    if not chunks:
        return {"status": "error", "message": "Failed to chunk document"}

    file_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
    for idx, ch in enumerate(chunks):
        ch["point_id"] = _deterministic_point_id(f"{session_id}_{meta['filename']}_{file_hash}", idx)
        ch["chunk_index"] = idx

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    points = [
        rest.PointStruct(id=c["point_id"], vector=vec, payload=c)
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=USER_DOCS_COLLECTION, points=points, wait=True)

    print(f"[INFO] Incremental upsert of {len(chunks)} chunks into '{USER_DOCS_COLLECTION}' for session {session_id}.")
    return {
        "status": "success",
        "collection": USER_DOCS_COLLECTION,
        "chunks_ingested": len(chunks),
        "filename": file_path.name
    }


def main(data_dir: str = "data") -> None:
    """Entry point for statutory ingestion CLI."""
    ingest_statutory_corpus(data_dir=data_dir, force_recreate=True)


if __name__ == "__main__":
    main()
