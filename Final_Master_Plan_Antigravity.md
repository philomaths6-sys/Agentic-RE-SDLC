# Final Master Plan: Agentic Requirement Engineering & SDLC Advisor
### 6-agent, fully-local, Postgres + Qdrant + FastAPI + Streamlit, autonomous Antigravity build

---

## 1. Domain — Finance (back to the original problem statement's domain)

This now follows the docx directly: financial institutions, heavy regulation, security-sensitive customer data, multiple stakeholder types (compliance officers, risk teams, auditors), audit/traceability requirements. It has genuinely public, well-structured regulatory documents to ground your RAG knowledge base in (Section 3) — real regulator-published text, not summaries you'd have to paraphrase from paywalled sources.

Example scenarios to test against (Antigravity will ask you to confirm/refine these, but start here):
1. Digital banking customer onboarding with KYC/identity verification
2. Loan origination and credit assessment workflow
3. Real-time payment processing with fraud-detection checks
4. Regulatory reporting pipeline for a mid-size bank

---

## 2. Agent architecture — 1 Supervisor + 5 worker agents

```
                         ┌──────────────────────────────┐
   User (Streamlit) ────▶│   Supervisor / Planning Agent │
                         │  - decomposes the request      │
                         │  - decides agent execution      │
                         │    order & shared context       │
                         │  - routes low-confidence/       │
                         │    flagged output to human      │
                         │    approval                     │
                         └───────────────┬────────────────┘
                                          │
        ┌─────────────┬──────────────────┼──────────────────┬─────────────┐
        ▼             ▼                  ▼                  ▼             ▼
 ┌─────────────┐ ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ ┌──────────────┐
 │ 1. Elicita-  │ │ 2. Extraction│  │ 3. Quality & │  │ 4. SDLC      │ │ 5. Artefact & │
 │ tion Agent   │▶│ & Classifi-  │▶│ Compliance   │▶│ Recommenda-  │▶│ Traceability  │
 │ (adaptive    │ │ cation Agent │  │ Agent (RAG-  │  │ tion Agent   │ │ Generation    │
 │ interview)   │ │ (structures  │  │ grounded reg │  │ (ranked SDLC │ │ Agent (SRS,   │
 │              │ │ requirements)│  │ mapping,     │  │ + workflow)  │ │ user stories, │
 │              │ │              │  │ ambiguity/   │  │              │ │ traceability  │
 │              │ │              │  │ risk checks) │  │              │ │ matrix)       │
 └─────────────┘ └─────────────┘  └──────────────┘  └──────────────┘ └──────────────┘
```

This gives you 6 agents total — matches "4-5 agents plus a supervisor" and covers requirement gathering (agents 1-2), requirement analysis/compliance (agent 3), SDLC selection (agent 4), and artefact generation (agent 5) — i.e. most of the pipeline in the original docx, minus full risk-register/audit-log automation which stays a documented future extension.

### Fine-tuning scope: one base model, only 1-2 agents get a fine-tuned adapter

Per your call — don't fine-tune all six agents. **One shared base model (Llama-3.2-3B-Instruct, 4-bit) loaded once**, used by every agent via prompting. Only two agents get a dedicated LoRA adapter on top of that same base model, hot-swapped in at call time:

- **Extraction & Classification Agent** — benefits most from fine-tuning because its output must reliably hit a strict structured schema (requirement ID, category, priority, etc.) every time; prompting alone tends to drift in formatting over many calls.
- **SDLC Recommendation Agent** — benefits because ranking + justification style is exactly the kind of consistent, structured output LoRA is good at locking in.

The other four agents (Supervisor, Elicitation, Quality & Compliance, Artefact & Traceability) run on the **same base model with just prompting/few-shot examples** — no adapter, no separate training run. This is not a compromise: Elicitation is inherently conversational (fine-tuning tends to make it more rigid, not better), Compliance leans on RAG grounding rather than style, Artefact generation is templated output that few-shot prompting handles fine, and the Supervisor is pure routing logic.

Net result: 1 base model + 2 small LoRA adapters (~50-200MB each) — comfortably within 6GB, and a much smaller, more defensible fine-tuning scope for your evaluation write-up than "fine-tuned all six agents."

---

## 3. RAG knowledge base — real public sources to download

| Source | What it gives you | Link |
|---|---|---|
| PCI DSS (Payment Card Industry Data Security Standard) | Core security/compliance grounding for card-data and payment-processing requirements | https://www.pcisecuritystandards.org/standards/pci-dss/ |
| FFIEC IT Examination Handbook — Information Security booklet | Detailed, structured control-by-control guidance for US financial institutions — excellent RAG chunking material | https://ithandbook.ffiec.gov/ |
| GLBA Safeguards Rule (FTC, 16 CFR Part 314) | Privacy/data-protection requirement mapping for customer financial information | search "FTC Safeguards Rule 16 CFR 314" on ftc.gov |
| OWASP ASVS (Application Security Verification Standard) | Security requirement templates/checklist, maps well to your "security requirements" category | https://github.com/OWASP/ASVS |
| NIST SP 800-218 (Secure Software Development Framework) | Gives you real SDLC/DevSecOps phase and control language for the SDLC Agent's workflow generation | search "NIST SP 800-218 SSDF PDF" on nist.gov (CSRC publications page) |
| Sample public SRS documents | Format/structure examples to few-shot from | search GitHub for "software-requirements-specification example" — many university course repos publish full sample SRS PDFs/docs you can legally reuse for structure reference |

Download these into `data/knowledge_base/` as PDFs/text; Antigravity's ingestion script chunks and embeds them into Qdrant. All are public-domain government/regulator publications or open-license standards — safe to use directly, no scraping paywalled/copyrighted material needed.

---

## 3.5 Data folder structure — where everything goes

Antigravity should scaffold this exact structure in Stage 1, before anything else, so there's one obvious place to drop each kind of file:

```
data/
├── knowledge_base/          ← DROP YOUR RAG SOURCE DOCS HERE
│   ├── pci_dss/              (PCI DSS PDF from Section 3)
│   ├── ffiec/                (FFIEC IT Handbook booklets)
│   ├── glba/                 (GLBA Safeguards Rule text)
│   ├── owasp_asvs/           (OWASP ASVS PDF/markdown)
│   └── nist_ssdf/            (NIST SP 800-218 PDF)
├── raw_scenarios/            ← drop the 4 example scenarios (Section 1) here as
│                               plain .txt/.md files, one per scenario
├── sample_srs/               ← optional: drop 2-3 public sample SRS documents
│                               here for the Artefact Agent to few-shot from
├── priority_reference/       ← drop a short .md file describing MoSCoW rules
│                               (Section "prioritization" below) — the Extraction
│                               Agent's prompt references this file directly
└── training/                 ← auto-generated by Stage 6, don't put anything
    ├── agent_extraction.jsonl   here manually
    └── agent_sdlc.jsonl
```

**Practical rule:** anything that should ground an agent's answer with a citation (regulations, standards) goes in `knowledge_base/` and gets embedded into Qdrant. Anything that's a formatting/style example (sample SRS, MoSCoW rules) goes in `sample_srs/` or `priority_reference/` and is used as few-shot prompt content, not retrieval — it's small enough to just include directly rather than round-tripping through the vector DB.

---

## 3.6 Requirement prioritization — MoSCoW, made explicit

Don't leave "priority" to the LLM's judgment alone. Put a short rules file at `data/priority_reference/moscow_rules.md` that the Extraction & Classification Agent's prompt explicitly references:

- **Must Have** — the system is non-functional, non-compliant, or unsafe without it. Auto-applies to anything the Compliance Agent maps to a PCI-DSS/FFIEC/GLBA control, and to any core transactional flow (e.g., authentication before a payment).
- **Should Have** — important but the system works without it short-term; a workaround exists.
- **Could Have** — desirable, low impact if left out this cycle.
- **Won't Have (this cycle)** — explicitly deferred, recorded so it isn't silently lost.

Wire this into the pipeline as a rule, not a suggestion: **any requirement the Compliance Agent links to a regulatory control gets its priority auto-set to "Must Have"** regardless of what the Extraction Agent guessed — compliance-driven requirements shouldn't be down-graded by a language model's judgment call.

---

## 4. Native infra install (no Docker) — Linux

**PostgreSQL:**
```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable postgresql --now
sudo -u postgres psql -c "CREATE USER agentic_user WITH PASSWORD 'changeme';"
sudo -u postgres psql -c "CREATE DATABASE agentic_re_sdlc OWNER agentic_user;"
```

**Qdrant (no Docker — prebuilt binary):**
```bash
# Check the latest release tag at https://github.com/qdrant/qdrant/releases first
curl -L -o qdrant.tar.gz https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar xzf qdrant.tar.gz
./qdrant --config-path config/config.yaml &
# Runs on localhost:6333 by default (REST) / 6334 (gRPC)
```
If the prebuilt binary doesn't match your distro/glibc version, the fallback is building from source with `cargo build --release` (needs Rust installed) — Antigravity should try the binary first and only fall back to source build if it fails.

---

## 5. Backend/Frontend

- **FastAPI** backend exposing: `/session` (create/get), `/session/{id}/message` (chat turn, routes through Supervisor), `/requirements/{session_id}` (structured output), `/sdlc/{session_id}` (recommendation), `/approve/{item_id}` (human-in-the-loop action), `/artefacts/{session_id}` (SRS/user-stories export).
- Postgres via SQLAlchemy + Alembic migrations — tables for sessions, requirements, sdlc_recommendations, approvals, audit_log.
- Qdrant accessed via `qdrant-client` from within the FastAPI backend (RAG retrieval happens server-side, not in Streamlit).
- **Streamlit** frontend is a thin chat client that only talks to the FastAPI backend over HTTP — no business logic in Streamlit itself. This separation matters: it's what makes this look like a real system rather than a notebook demo, and it's a legitimate design point to mention in your evaluation.

---

## 6. Master prompt for Antigravity — full autonomous build

Paste this whole block as your task. It's written to let Antigravity self-verify each stage (check GPU, check services are running, run a smoke test) before moving to the next, since you asked it to do everything — including fine-tuning and RAG — on its own.

```
Build a fully local, offline multi-agent Agentic AI system called "agentic-re-sdlc"
for automated software requirement engineering and SDLC recommendation in the
financial services domain (digital banking / payments platform), running entirely
on Linux with an RTX 3050 (6GB VRAM). Do the full build end-to-end, including local
fine-tuning and RAG ingestion — ask me only when you hit a genuine ambiguity or a
step fails, otherwise proceed autonomously through the stages below and report
progress after each one.

IMPORTANT SCOPE NOTE ON FINE-TUNING: only fine-tune the Extraction & Classification
Agent and the SDLC Recommendation Agent. All other agents (Supervisor, Elicitation,
Quality & Compliance, Artefact & Traceability) must run on the same shared base
model using prompting/few-shot examples only — do not train adapters for them, and
do not treat this as a shortcut to skip later; it's the intended final scope.

STAGE 0 — ENVIRONMENT CHECK
Run nvidia-smi and confirm ~6GB VRAM is available; if not, pause and tell me the
actual number before proceeding, since model/batch size choices below depend on it.
Confirm Python 3.10+, pip, and enough disk space (~15GB for models + deps) are
available.

STAGE 1 — INFRA + DATA FOLDER SCAFFOLD (native, no Docker)
- Create the data/ folder structure exactly as follows, with a README.md inside
  data/ explaining what goes where:
  data/knowledge_base/{pci_dss,ffiec,glba,owasp_asvs,nist_ssdf}/  (RAG source docs)
  data/raw_scenarios/            (example scenario .txt/.md files)
  data/sample_srs/               (optional public sample SRS docs for few-shot)
  data/priority_reference/       (MoSCoW rules file, see below)
  data/training/                 (auto-generated later by Stage 6, leave empty)
- Write data/priority_reference/moscow_rules.md defining: Must Have (regulatory/
  compliance-linked or core-transactional, non-negotiable), Should Have (important,
  workaround exists), Could Have (low impact if deferred), Won't Have this cycle
  (explicitly deferred, not lost). This file is loaded into the Extraction &
  Classification Agent's prompt directly (few-shot, not RAG-retrieved).
- Install and start PostgreSQL natively; create database "agentic_re_sdlc" and user
  "agentic_user". Write db/schema.sql with tables: sessions, requirements (id,
  session_id, statement, category[] multi-label, source, priority, dependencies,
  acceptance_criteria, applicable_regulations, confidence_score, approval_status),
  sdlc_recommendations (session_id, model_name, confidence_pct, justification,
  workflow_json), approvals (item_type, item_id, action, timestamp, actor),
  audit_log (actor, action, payload, timestamp). Use SQLAlchemy models + Alembic
  migrations.
- Download the latest Qdrant release binary for linux-x86_64 (fall back to
  `cargo build --release` from source if the prebuilt binary fails on this glibc
  version), run it locally on default ports, and confirm it responds on
  localhost:6333/healthz before continuing.

STAGE 2 — RAG KNOWLEDGE BASE
- Download these public documents into their matching data/knowledge_base/
  subfolder from Stage 1:
  - PCI DSS standard → data/knowledge_base/pci_dss/ (pcisecuritystandards.org/standards/pci-dss)
  - FFIEC IT Examination Handbook, Information Security booklet → data/knowledge_base/ffiec/ (ithandbook.ffiec.gov)
  - GLBA Safeguards Rule / FTC 16 CFR Part 314 → data/knowledge_base/glba/ (ftc.gov)
  - OWASP ASVS latest release → data/knowledge_base/owasp_asvs/ (github.com/OWASP/ASVS)
  - NIST SP 800-218 SSDF → data/knowledge_base/nist_ssdf/ (search nist.gov CSRC for the current PDF)
  If I've already dropped files into any of these folders myself before you run
  this stage, use those instead of re-downloading.
- Build rag/ingest.py: chunk these (semantic or fixed-size ~500 tokens with overlap),
  embed with BAAI/bge-small-en-v1.5 via sentence-transformers (CPU is fine), upsert
  into a Qdrant collection "finance_compliance_kb" with source metadata
  (document name, section, url) attached to every chunk for citation.
- Build rag/retriever.py with a top-k similarity search function callable by any
  agent.

STAGE 3 — GUARDRAILS
- guardrails/input_filter.py: regex/NER PII masking (names, account numbers, card
  numbers, SSNs, emails, phone numbers) applied to any text before it reaches an
  LLM call; a prompt-injection heuristic that flags/rejects inputs containing
  instruction-override patterns.
- guardrails/output_filter.py: run Llama-Guard-3-1B (4-bit, CPU or shared GPU slot)
  over every agent's generated output; also apply a confidence threshold below
  which output is routed to the approvals table as "needs_review" instead of being
  auto-finalized.

STAGE 4 — AGENTS (baseline, prompted, no fine-tuning yet)
Implement as LangGraph nodes, one shared base model instance
(Llama-3.2-3B-Instruct, 4-bit via Unsloth) used by all agents:
1. Supervisor/Planning Agent (agents/supervisor.py) — decomposes the user's message,
   decides which downstream agent(s) to invoke and in what order, holds shared
   session state, and is the only agent allowed to trigger the human-approval
   interrupt node. Keep this mostly rule-based/lightweight-LLM routing, not a full
   generation call, for reliability.
2. Elicitation Agent (agents/elicitation_agent.py) — adaptive interview: asks
   follow-up questions when input is vague/incomplete, referencing the scenario
   categories from data/knowledge_base and the example scenarios I'll provide.
   Prompted only, no fine-tuning.
3. Extraction & Classification Agent (agents/extraction_agent.py) — converts the
   elicited conversation into the structured requirement schema from db/schema.sql,
   multi-label category classification allowed. Assign priority using the MoSCoW
   rules in data/priority_reference/moscow_rules.md (few-shot in the prompt, not
   RAG-retrieved). This is one of the two agents that gets a fine-tuned LoRA
   adapter (Stage 6).
4. Quality & Compliance Agent (agents/compliance_agent.py) — checks each requirement
   for ambiguity/incompleteness/inconsistency/duplication, and uses the RAG
   retriever to map requirements to PCI-DSS/FFIEC/GLBA/OWASP controls with cited
   sources; flags high-risk items for mandatory human approval rather than
   auto-approving. Any requirement this agent links to a regulatory control must
   have its priority force-set to "Must Have" in the requirements table, overriding
   whatever the Extraction Agent assigned — compliance-linked priority is a rule,
   not a model judgment call. Prompted + RAG only, no fine-tuning.
5. SDLC Recommendation Agent (agents/sdlc_agent.py) — derives project
   characteristics (regulatory criticality, requirement stability, security risk,
   complexity, legacy dependence, change frequency) from the structured
   requirements, and outputs ranked SDLC options (e.g. Agile-DevSecOps hybrid,
   V-Model, Waterfall, Spiral) with percentage confidence and a justification tied
   to the specific factors found, plus a phase-by-phase workflow (activities,
   roles, deliverables, validation gates, security checkpoints, human-approval
   gates). This is the second agent that gets a fine-tuned LoRA adapter (Stage 6).
6. Artefact & Traceability Agent (agents/artefact_agent.py) — generates an SRS-style
   document, user stories with acceptance criteria, and a traceability matrix
   linking every artefact back to its originating requirement and, where
   applicable, the elicitation conversation turn that produced it. Persist all of
   this to Postgres. Prompted + few-shot only, no fine-tuning.

STAGE 5 — BACKEND + FRONTEND
- FastAPI backend (backend/main.py) exposing: POST /session, POST
  /session/{id}/message (routes through Supervisor and returns the active agent's
  response plus any pending approvals), GET /requirements/{session_id}, GET
  /sdlc/{session_id}, POST /approve/{item_id}, GET /artefacts/{session_id}. All
  business logic lives here — Postgres and Qdrant access happen only from the
  backend.
- Streamlit frontend (ui/app.py): a chat interface that calls the FastAPI backend
  over HTTP only (no direct DB/model access from Streamlit). Show the live
  conversation, a table view of structured requirements as they're produced, the
  SDLC recommendation with confidence bars, and approve/edit/reject buttons for
  anything flagged "needs_review".
- Run a smoke test end-to-end (one of the example scenarios) through the full
  stack before moving to fine-tuning, and show me the output.

STAGE 6 — FINE-TUNING (local QLoRA, only 2 agents, after baseline works)
- finetune/prepare_dataset.py: generate synthetic instruction-tuning datasets
  (JSONL), one for the Extraction & Classification Agent and one for the SDLC
  Recommendation Agent — 300-800 examples each, using whatever strong model I have
  API access to for generation (ask me which if none is configured), covering the
  four example financial scenarios plus variations. Do NOT generate training data
  for Elicitation, Compliance, Artefact, or Supervisor — they stay prompted-only.
- finetune/train_lora.py using Unsloth + PEFT: base Llama-3.2-3B-Instruct, 4-bit,
  LoRA rank 16 / alpha 32 / dropout 0.05, target modules q/k/v/o_proj, batch size
  2 with gradient accumulation 4, 3 epochs, max seq length 1024. Train exactly two
  adapters: one for Extraction & Classification, one for SDLC Recommendation. Keep
  VRAM usage under 6GB — reduce batch size / sequence length automatically if you
  hit OOM, don't just fail.
- finetune/eval_holdout.py: compare base-model vs. fine-tuned-adapter outputs on a
  held-out 15-20% split — report structure adherence, category-classification
  accuracy, and SDLC-recommendation agreement with the synthetic gold labels.
- Wire the trained adapters into agents/extraction_agent.py and agents/sdlc_agent.py
  with a fallback to base-model prompting if an adapter fails to load.

STAGE 7 — WRAP-UP
Write README.md documenting: architecture diagram, which 2 of the 6 agents are
fine-tuned and why the other 4 are prompted-only, how to start Postgres/Qdrant/
backend/frontend, how to re-run fine-tuning, what's implemented vs. explicitly
out-of-scope (full audit-trail automation, multi-jurisdiction regulation beyond
US financial regulations, risk-register automation), and the evaluation results
from Stage 6.

Confirm with me before Stage 6 if Stage 4-5 took longer than expected, since
fine-tuning is the part most sensitive to running out of time — I'd rather have a
solid Stage 5 baseline demoed than a half-finished Stage 6.
```

---

## 7. What to expect realistically

Stages 0-5 (infra, RAG, guardrails, 6 prompted agents, FastAPI+Streamlit, smoke test) are very achievable in one focused push and give you a fully working, demoable system on their own — this is already more than the 2-agent minimum, and all 6 agents work end-to-end even before any fine-tuning happens. Stage 6 (fine-tuning, scoped to just 2 agents) is the part most likely to eat unplanned time (dataset generation quality, OOM debugging, adapter integration bugs) — the prompt explicitly tells Antigravity to check in with you before starting it if earlier stages ran long, so you're not stuck mid-fine-tune the night before a deadline with no working demo at all.
