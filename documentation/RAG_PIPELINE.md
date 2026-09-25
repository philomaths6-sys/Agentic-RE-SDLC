# Statutory Compliance RAG Pipeline

> Deep dive into the **Qdrant Vector Retrieval Architecture** powering compliance grounding in the Agentic RE-SDLC platform.

---

## 1. Knowledge Base Corpus

The RAG pipeline grounds all agent decisions in sovereign Indian financial regulations stored in `data/knowledge_base/` and `data/indian_docx/`.

### Statutory Coverage

| Regulatory Authority | Frameworks / Circulars Ingested | Chunks Active |
|---|---|:---:|
| **Reserve Bank of India (RBI)** | Master Direction on Digital Payment Security Controls (2021), Cyber Security Framework for Banks, Storage of Payment System Data (2018), KYC Master Direction (V-CIP Amendment), Harmonisation of Turn Around Time (TAT) & Customer Compensation (2019) | 1,420 |
| **National Payments Corp of India (NPCI)** | UPI Procedural Guidelines, Mobile Application Security Standards v2.4, UPI AutoPay Recurring Mandate Specifications, Tokenisation Directives | 680 |
| **Securities and Exchange Board of India (SEBI)** | Cybersecurity and Cyber Resilience Framework (CSCRF 2024), System Audit Norms, Algorithmic Trading Security Standards | 440 |
| **Unique Identification Authority of India (UIDAI)** | Aadhaar Data Vault Circular (2017), Aadhaar Masking Regulations, e-KYC Security Controls | 290 |
| **Digital Personal Data Protection (DPDP)** | DPDP Act 2023 Statutory Sections (Consent Architecture, Data Principal Rights, 72-Hour Erasure, Cross-Border Transfer Norms) | 289 |
| **Total Active Chunks** | — | **3,119** |

---

## 2. Ingestion & Chunking Strategy

Documents (`.docx`, `.pdf`, `.txt`) are processed via [`rag/indexer.py`](../rag/indexer.py):

```
 Raw Document (.docx / .pdf)
             │
             ▼
 Document Parser (python-docx / pypdf)
             │
             ▼
 Semantic Section Splitter (Header / Sub-clause aware)
             │  Chunk size: 400–600 tokens
             │  Overlap: 50 tokens
             ▼
 Sentence-Transformers Embedding Engine (all-MiniLM-L6-v2)
             │  Output: 384-dimensional dense vectors
             ▼
 Qdrant Vector DB Ingestion (Collection: 'documents')
             │  Distance metric: Cosine Similarity
             ▼
 On-Disk / Memory Inverted HNSW Index
```

### Metadata Payload Schema
Each vector point in Qdrant contains rich structured metadata:
```json
{
  "id": "chunk_rbi_sec_4_3_104",
  "citation": "RBI Digital Payment Security Controls Sec 4.3",
  "authority": "RBI",
  "text": "The mobile application shall enforce device fingerprint binding, automated background biometric re-authentication prompts, and visual privacy watermarks on all sensitive screens displaying customer account numbers...",
  "category": "Mobile Security & Device Binding",
  "effective_date": "2021-02-18",
  "mandatory": true
}
```

---

## 3. Retrieval Pipeline (`rag/retriever.py`)

When the multi-agent swarm processes a project requirement:

1. **Query Synthesis**: The Elicitation Agent generates targeted search queries (e.g. `device binding biometric authentication mobile app RBI security controls`).
2. **Dense Vector Search**: The query is converted to a 384-dimensional vector and executed against Qdrant via cosine distance.
3. **Threshold Filtering**: Results with cosine similarity below `0.65` are filtered out to prevent hallucinations.
4. **Context Formatting**: Top-K retrieved chunks (default `k=5`) are formatted with statutory citations and injected into the Extraction Agent's context window.

```python
# rag/retriever.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    vector = model.encode(query).tolist()
    hits = client.search(
        collection_name="documents",
        query_vector=vector,
        limit=top_k
    )
    return [{"text": h.payload["text"], "citation": h.payload["citation"]} for h in hits]
```
