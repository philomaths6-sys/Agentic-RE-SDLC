# Agentic RE-SDLC — Sovereign Requirements Engineering & Adaptive Life Cycle Governance

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-brightgreen.svg)]()
[![Framework: FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal.svg)]()
[![Model: Qwen2.5-3B-Instruct](https://img.shields.io/badge/Model-Qwen2.5--3B--4bit-orange.svg)]()
[![Vector DB: Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red.svg)]()

> An autonomous multi-agent engineering platform designed for regulated Indian financial entities (RBI, NPCI, UIDAI, SEBI, DPDP Act 2023) — decomposing unstructured project descriptions into **30+ modular specifications across 6 architectural pillars**, grounding every clause via high-speed statutory vector RAG, and predicting adaptive SDLC execution models via a fine-tuned 4-bit LoRA adapter with human-in-the-loop stage gates.

---

## The Problem

In regulated financial software engineering (banking, UPI payments, capital markets, insurance), building software without statutory compliance is a recipe for catastrophic regulatory fines or operational shutdown. Indian institutions must strictly satisfy hundreds of overlapping mandates from the **Reserve Bank of India (RBI)**, **NPCI**, **SEBI**, **UIDAI**, and the **DPDP Act 2023**:

- Requirements engineering is manual, slow, and error-prone — taking enterprise teams 8 to 12 weeks to draft a single compliant SRS.
- Off-the-shelf generative AI models hallucinate generic Silicon Valley patterns that fail Indian statutory criteria (e.g. failing to specify Aadhaar Data Vault HSM isolation or violating RBI 100% on-soil data residency).
- Most software teams default blindly to "Scrum Agile" or "Waterfall" without evaluating statutory liability — leading to delayed regulatory audits or non-compliant production releases.

The gap no existing tool fills:

| Tool / Approach | Approach | Limitation |
|---|---|---|
| **Generic LLMs (ChatGPT / Claude)** | Generates high-level, unstructured bullet points | Zero awareness of Indian statutory mandates (RBI/SEBI/DPDP); hallucinations; lacks measurable acceptance criteria |
| **Traditional Manual RE** | Business analysts draft 100-page static Word documents | 8–12 week turnaround; specifications become obsolete when circulars change mid-flight; poor developer traceability |
| **Standard Agile Sprints** | Assumes uniform velocity and flexible release scope | Fails statutory stage gates (e.g. CERT-In audit required before core banking cutover); risks severe RBI penalties |
| **Agentic RE-SDLC** | Autonomous 6-agent swarm with statutory RAG & LoRA SDLC | **Delivers 30+ modular specs in 12s**, 100% pillar coverage, automated SRS PDF generation, and adaptive SDLC prediction |

---

## What Agentic RE-SDLC Does

The platform runs an autonomous multi-agent swarm that turns natural-language project descriptions or uploaded `.docx`/`.pdf` documents into a production-grade, audit-ready engineering specification:

1. **Defends the Perimeter**: Scans all incoming queries with Zero-Trust guardrails — blocking prompt injection and redacting unmasked PII (Aadhaar, PAN, passwords).
2. **Profiles Intent & Risk**: Automatically classifies business intent, scores regulatory risk on a 0–100 scale, and establishes MoSCoW baselines.
3. **Retrieves Statutory Grounding**: Queries 3,119 active chunks in Qdrant Vector DB across RBI Master Directions, NPCI guidelines, SEBI CSCRF, and DPDP Act 2023.
4. **Decomposes into 6 Architectural Pillars**: Extracts at least 30 modular specifications spanning Frontend UI/UX, Backend APIs, Auth & KYC, Security, Database Residency, and Resilience.
5. **Compiles Sovereign SRS Reports**: Generates formal, downloadable PDF reports with executive risk scoring, traceability matrices (RTM), and acceptance criteria.
6. **Enforces Human-in-the-Loop Gate 1**: Freezes execution until the human architect reviews, edits, and approves the generated requirements.
7. **Predicts Adaptive SDLC Models**: Invokes a fine-tuned 4-bit LoRA model (`unsloth/Qwen2.5-3B-Instruct`) to recommend the optimal SDLC model out of 10 methodologies with full regulatory justifications and alternatives evaluation.
8. **Enforces Human-in-the-Loop Gate 2**: Allows the architect to accept or override the recommended model, dynamically updating the WBS, sprint backlog, and milestone roadmap.

---

## Architecture

```
   /session/message              /sdlc/recommend
  (User Query / SRS)          (Stage-Gate Review)
          │                            │
  guardrails/input_filter.py    agents/sdlc_agent.py
  (Zero-Trust Gatekeeper)       (4-bit LoRA Adapter)
          │                            │
          └─────────── backend/main.py ────────┘
                    (FastAPI Orchestrator)
                              │
  ┌───────────────────────────┴───────────────────────────┐
  ▼                                                       ▼
rag/retriever.py                                requirements_analysis.py
(Qdrant Vector DB 6333)                         (6-Pillar Extraction Engine)
  │                                                       │
  └───────────────────────────┬───────────────────────────┘
                              ▼
                       pdf_generator.py
                 (Sovereign SRS PDF Compiler)
                              ▼
                         ui/app.py
                  (Streamlit Dark Mode UI)
```

### Components

| Component | File | Responsibility |
|---|---|---|
| **Supervisor Agent** | [`guardrails/input_filter.py`](guardrails/input_filter.py) | Scans for adversarial prompts, SQLi delimiters, and masks raw PII (Aadhaar, PAN, Passwords). |
| **Elicitation Agent** | [`agents/requirement_generator.py`](agents/requirement_generator.py) | Classifies domain intent, calculates statutory risk (0–100), and assigns MoSCoW priority baselines. |
| **Compliance RAG Agent** | [`rag/retriever.py`](rag/retriever.py) | Executes dense vector searches on Qdrant across 3,119 indexed statutory circulars. |
| **Architecture Extractor** | [`agents/extraction_agent.py`](agents/extraction_agent.py) | Generates 30+ modular specifications across 6 pillars with measurable acceptance criteria. |
| **Artefacts Generator** | [`pdf_generator.py`](pdf_generator.py) | Compiles executive-ready PDF SRS documents with color-coded badges and traceability tables. |
| **SDLC Governance Agent** | [`agents/sdlc_agent.py`](agents/sdlc_agent.py) | Fine-tuned 4-bit LoRA adapter evaluating 10 SDLC models against statutory governance constraints. |
| **Model Manager** | [`agents/model_manager.py`](agents/model_manager.py) | Caches local Qwen-2.5-3B base model in GPU VRAM and hot-swaps LoRA adapters on demand. |
| **Database Persistence** | [`db/database.py`](db/database.py) | Multi-session storage in PostgreSQL 16 (sessions, requirements, approvals, audit logs). |

### Multi-Agent Lifecycle State Machine

```
  ┌────────────────────────────────────────────────────────┐
  │  STATE_IDLE                                            │
  │  Awaiting user query or document upload.               │
  │  → DISPATCHING upon receipt of input payload           │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │  STATE_DISPATCHING                                     │
  │  Supervisor scans injection & redacts PII.             │
  │  Elicitation scores risk (0-100) & sets MoSCoW.        │
  │  → RETRIEVING if safe; ABORT if malicious              │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │  STATE_RETRIEVING                                      │
  │  Qdrant dense vector search (top-k=5, sim > 0.65).     │
  │  Grounds RBI, SEBI, NPCI, UIDAI, DPDP circulars.       │
  │  → EXTRACTING upon context injection                   │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │  STATE_EXTRACTING                                      │
  │  Decomposes into 6 pillars (30+ modular requirements). │
  │  Compiles downloadable SRS PDF report.                 │
  │  → AWAITING_GATE_1 (Human Verification Lock)           │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │  STATE_AWAITING_GATE_1 (Human-in-the-Loop)             │
  │  System locks execution. Architect inspects/edits.     │
  │  → PREDICTING_SDLC on human verification & approval    │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │  STATE_PREDICTING_SDLC                                 │
  │  LoRA adapter predicts optimal SDLC model & rationale. │
  │  Evaluates alternatives and explains trade-offs.       │
  │  → AWAITING_GATE_2 (Model Approval / Override)         │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │  STATE_AWAITING_GATE_2 (Human-in-the-Loop)             │
  │  Architect accepts recommendation or selects override. │
  │  Generates stage-gate roadmap, WBS, and milestones.    │
  │  → COMPLETED                                           │
  └────────────────────────────────────────────────────────┘
```

---

## Supported SDLC Models (10 Models)

The fine-tuned SDLC governance agent evaluates 10 distinct models based on operational risk, technical uncertainty, and statutory constraints:

| SDLC Model | Primary Applicability in Regulated Systems | Real-World Example |
|---|---|---|
| **Agile-DevSecOps Hybrid** | Dynamic consumer credit, digital lending (LSP), evolving regulations | UPI AutoPay Recurring Mandates Platform |
| **V-Model with DevSecOps Gates** | Direct capital risk, hardware security modules (HSM), automated trade routing | SEBI CSCRF Algorithmic Trading Gateway |
| **Incremental Model** | Multi-phase pilot programs requiring independent statutory certification | RBI Central Bank Digital Currency (CBDC / e-Rupee) |
| **Iterative Model** | Large legacy modernization with planned subsystem replacement | Core Banking COBOL-to-Microservices Migration |
| **Waterfall** | 100% frozen statutory formats with zero tolerance for scope drift | IRDAI Statutory Annual Return & Solvency Filings |
| **Spiral Model** | High compound uncertainty, experimental AI risk, or ML anomaly scoring | Real-time AI Fraud Detection Engine |
| **Prototyping Model** | Highly ambiguous UX, novel human-computer interaction, vernacular interfaces | Multilingual Voice-Activated Rural UPI Assistant |
| **Rapid Application Dev (RAD)** | Well-understood internal tools, non-PII dashboards, short 4–6 week timelines | Internal RBI CRILC/OSMOS Reporting Dashboard |
| **Dual-Track Agile** | Fast discovery sprints paired with continuous security deployment | Personalized WealthTech & Robo-Advisory App |
| **V-Model (Classic)** | Formal regulatory verification and static statutory testing matrices | Statutory Clearing House Settlement Engine |

---

## Key Design Decisions

### 1. 4-bit NF4 Quantization & Local Edge Execution
The model runs locally using `bitsandbytes` 4-bit NF4 quantization on an **RTX 3050 Laptop GPU (6.0 GB VRAM)**. This eliminates recurring cloud API inference costs, guarantees **100% on-soil data residency** (no prompt data ever leaves the host machine), and preserves data privacy.

### 2. LoRA Hot-Swapping Architecture
Base model weights (`Qwen2.5-3B-Instruct`, ~2.05 GB) are loaded once into GPU memory. Dedicated domain adapters (`finetune/adapters/sdlc_adapter`, ~29 MB) are hot-swapped dynamically into shared GPU memory without restarting or re-allocating VRAM.

### 3. Mandatory 6-Pillar Engineering Standard
Rather than generating arbitrary bullet points, every system requirement is decomposed into 6 non-negotiable architectural pillars (Frontend UI/UX, Backend APIs, Auth & KYC, Security, Database Residency, Resilience), ensuring that security and compliance are never treated as afterthoughts.

---

## Prerequisites

| Requirement | Specification |
|---|---|
| **Operating System** | Linux (Ubuntu 20.04/22.04/24.04 recommended) |
| **Python** | Python ≥ 3.10 |
| **GPU (Optional but Recommended)** | NVIDIA GPU with ≥ 4 GB VRAM (RTX 3050, T4, RTX 4060, etc.) |
| **Docker** | Docker & Docker Compose (for PostgreSQL and Qdrant) |

---

## Quickstart & Installation

### Step 1 — Clone Repository & Setup Environment

```bash
git clone https://github.com/philomaths6-sys/Agentic-RE-SDLC.git
cd Agentic-RE-SDLC

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Start Infrastructure (PostgreSQL & Qdrant)

```bash
# Start PostgreSQL container on port 5433
docker run -d --name pg_agentic -p 5433:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=agentic_db postgres:16-alpine

# Start Qdrant Vector DB on port 6333
./qdrant
```

### Step 3 — Launch Backend & Frontend Services

```bash
# Terminal 1 — Start FastAPI Backend (Port 8000)
env PYTHONPATH=. ./venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Start Streamlit Web UI (Port 8501)
env PYTHONPATH=. ./venv/bin/streamlit run ui/app.py --server.port 8501
```

Open your browser to `http://localhost:8501` to access the interactive web interface.

---

## Sample Output

Live recommendation output from the fine-tuned SDLC governance agent:

```
[RESULT] Recommended SDLC Model: ===>>> Incremental Model <<<===
[REASONING]: The Incremental Model is chosen because the digital rupee migration itself is
structured as an incremental sequence of smaller, independently deployable modules
(e-Rupee Wallet → P2P Transfer → Merchant QR Payment → Programmable CBDC), where each
module must be certified by RBI's CBDC pilot authority before the next one begins development.
Each increment has its own functional specification, delivery timeline, testing plan, and
regulatory certification checklist that makes sequential delivery the most efficient and
risk-controlled approach.
[ALTERNATIVES EVALUATED]:
  - Iterative Model: Rejected because Iterative revisits and refines the same functionality
    across cycles, whereas CBDC increments are distinct new capabilities.
  - Waterfall: Rejected because waiting for all increments to be coded before certifying
    violates RBI pilot milestones.
```

---

## Benchmark Results

| Test Case | Domain Focus | Pillar Coverage | Requirements | Recommended Model | End-to-End Latency |
|---|---|:---:|:---:|---|:---:|
| **TC-ATM-01** | ATM Cash Cassette Switch | 100% (6/6) | 33 Reqs | V-Model with DevSecOps Gates | 11.2s |
| **TC-UPI-02** | UPI AutoPay Mandate Switch | 100% (6/6) | 32 Reqs | Agile-DevSecOps Hybrid | 12.1s |
| **TC-TRD-03** | Algorithmic Trading Gateway | 100% (6/6) | 33 Reqs | V-Model with DevSecOps Gates | 11.8s |
| **TC-SBX-04** | Microfinance Rural Sandbox | 100% (6/6) | 30 Reqs | Iterative Model | 10.9s |

*For detailed benchmarks, see [`documentation/EVALUATION_AND_BENCHMARKS.md`](documentation/EVALUATION_AND_BENCHMARKS.md).*

---

## Documentation

Full architectural documentation is available in [`documentation/`](documentation/):

- [`documentation/ARCHITECTURE.md`](documentation/ARCHITECTURE.md) — System topology, 6-pillar framework, and hardware budgeting.
- [`documentation/AGENTS_AND_ORCHESTRATION.md`](documentation/AGENTS_AND_ORCHESTRATION.md) — Deep dive into all 6 autonomous specialist agents.
- [`documentation/RAG_PIPELINE.md`](documentation/RAG_PIPELINE.md) — Qdrant vector retrieval architecture and statutory corpus indexing.
- [`documentation/EVALUATION_AND_BENCHMARKS.md`](documentation/EVALUATION_AND_BENCHMARKS.md) — Comprehensive empirical benchmark reports.

---

## License

This project is licensed under the [MIT License](LICENSE).
