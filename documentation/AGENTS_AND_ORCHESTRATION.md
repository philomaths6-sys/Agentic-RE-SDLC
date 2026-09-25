# Autonomous Multi-Agent Swarm & Orchestration

> Detailed technical specifications of the **6 Autonomous Specialist Agents** powering the Agentic RE-SDLC platform.

---

## Agent Topology Overview

```
                      ┌───────────────────────────────┐
                      │     User Query / SRS Doc      │
                      └───────────────┬───────────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │      1. Supervisor Agent      │
                      │  - Prompt Injection Defense   │
                      │  - PII Masking / Redaction    │
                      │  - Domain Scope Validation    │
                      └───────────────┬───────────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │     2. Elicitation Agent      │
                      │  - Intent Classification      │
                      │  - Risk Scoring (0–100)       │
                      │  - MoSCoW Baseline Framing    │
                      └───────────────┬───────────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │    3. Compliance RAG Agent    │
                      │  - Qdrant Vector Retrieval    │
                      │  - Statutory Circular Matching│
                      │  - Relevance Re-Ranking       │
                      └───────────────┬───────────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │   4. Architecture Extractor   │
                      │  - 6-Pillar Decomposition     │
                      │  - 30+ Modular Requirements   │
                      │  - Acceptance Criteria Engine │
                      └───────────────┬───────────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │     5. Artefacts Agent        │
                      │  - Sovereign SRS PDF Report   │
                      │  - Traceability Matrix (RTM)  │
                      │  - Session DB Synchronization │
                      └───────────────┬───────────────┘
                                      ▼
                         [ Human Stage Gate 1 ]
                                      ▼
                      ┌───────────────────────────────┐
                      │   6. SDLC Governance Agent    │
                      │  - 4-bit Fine-Tuned LoRA      │
                      │  - 10 SDLC Model Evaluation   │
                      │  - Phase Breakdown & Roadmap  │
                      └───────────────────────────────┘
```

---

## Deep Dive: The 6 Autonomous Specialists

### 1. Supervisor Agent (Zero-Trust Guardrail Gatekeeper)
- **Role**: Perimeter defense, adversarial prompt injection detection, and sensitive PII filtering.
- **Implementation**: [`guardrails/input_filter.py`](../guardrails/input_filter.py) and [`guardrails/output_filter.py`](../guardrails/output_filter.py).
- **Execution Logic**:
  - Regex & heuristic scanning for delimiter injection (`Ignore previous instructions`, `System prompt leak`, etc.).
  - PII maskers for raw Aadhaar (12 digits), PAN, CVV, and plaintext passwords.
  - Domain validation: Verifies that the input pertains to financial engineering, banking, capital markets, or insurance.
  - Emits: `Safety Score: 1.0 (Zero-Trust)`, `Anomalies: 0`.

### 2. Elicitation Agent (Intent Profiler & Risk Evaluator)
- **Role**: Identifies the primary business intent, estimates statutory risk, and establishes MoSCoW prioritization.
- **Implementation**: [`agents/requirement_generator.py`](../agents/requirement_generator.py) & [`requirements_analysis.py`](../requirements_analysis.py).
- **Risk Scoring Engine**:
  - Calculates statutory risk on a 0–100 scale:
    - `0–39 (LOW)`: Internal reporting, static dashboards.
    - `40–69 (MEDIUM)`: Customer CRM, non-monetary account services.
    - `70–89 (HIGH)`: Digital lending, consumer payments, card tokenization.
    - `90–100 (CRITICAL)`: Algorithmic trading gateways, ATM switches, RTGS settlement, hardware HSM key vaults.

### 3. Compliance RAG Agent (Statutory Context Grounding)
- **Role**: Retrieves relevant statutory circulars, master directions, and security standards from the 3,119 chunk knowledge base.
- **Implementation**: [`rag/retriever.py`](../rag/retriever.py) & [`rag/indexer.py`](../rag/indexer.py).
- **Grounding Scope**:
  - RBI Master Directions (Cyber Security, KYC, Digital Payments).
  - NPCI UPI Procedural Guidelines & Mobile Security Standards.
  - UIDAI Aadhaar Data Vault Circulars.
  - SEBI Cybersecurity and Cyber Resilience Framework (CSCRF).
  - Digital Personal Data Protection Act, 2023 (DPDP).

### 4. Architecture Extraction Agent (6-Pillar Decomposer)
- **Role**: Synthesizes the project requirement and RAG citations into 30+ modular specifications across all 6 architectural pillars.
- **Implementation**: [`agents/extraction_agent.py`](../agents/extraction_agent.py) & [`requirements_analysis.py`](../requirements_analysis.py).
- **Requirements Output Format**:
  - `REQ-ID`: Unique traceable ID (e.g. `REQ-BE-01`, `REQ-SEC-02`).
  - `Title`: Precise engineering capability.
  - `Statement`: Formal requirement specification.
  - `Priority`: `Must Have`, `Should Have`, or `Could Have`.
  - `Regulation`: Exact statutory citation (e.g. `RBI Cyber Security Framework Sec 5.2`).
  - `Acceptance Criteria`: Measurable latency, security, or throughput test criteria.

### 5. Artefacts & Traceability Agent (SRS Document Engine)
- **Role**: Compiles the 30+ specifications into a structured, production-grade PDF Software Requirements Specification (SRS) report.
- **Implementation**: [`pdf_generator.py`](../pdf_generator.py).
- **Document Features**:
  - Executive Prioritization & Statutory Risk Assessment Header.
  - Multi-Agent System Execution & Reasoning Trace.
  - Color-coded MoSCoW Priority badges.
  - Full Requirements Traceability Matrix (RTM).
  - Statutory compliance cross-references.

### 6. SDLC Governance Agent (Adaptive Life Cycle Advisor)
- **Role**: Predicts the optimal Software Development Life Cycle model, justifies the selection against regulatory constraints, and rejects unsuitable alternatives.
- **Implementation**: [`agents/sdlc_agent.py`](../agents/sdlc_agent.py) & [`finetune/adapters/sdlc_adapter`](../finetune/adapters/sdlc_adapter).
- **Model Support (10 Models)**:
  - `Agile-DevSecOps Hybrid`
  - `V-Model with DevSecOps Gates`
  - `V-Model (Classic)`
  - `Waterfall`
  - `Spiral Model`
  - `Prototyping Model`
  - `Iterative Model`
  - `Rapid Application Development (RAD)`
  - `Incremental Model`
  - `Dual-Track Agile`
- **Output Schema**:
  ```json
  {
    "recommended_model": "Incremental Model",
    "reasoning": "Detailed justification connecting architectural risk and statutory delivery phases.",
    "alternatives_considered": [
      {
        "model": "Waterfall",
        "why_not": "Rejected due to multi-phase pilot certification constraints."
      }
    ]
  }
  ```
