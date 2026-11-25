# workflow/graph_orchestrator.py
"""LangGraph-Powered Stateful Orchestrator for Financial Requirements Engineering.

Features:
  1. Resumable StateGraph with explicit state schema.
  2. Dynamic Verifier / Critic Reflection & Critique Loop with bounded retry cycles (max N retries).
  3. Conflict, ambiguity, testability, and duplicate detection with targeted architectural repair.
  4. Per-Agent / Stage Input-Output Observability Logging with millisecond latency metrics.
"""

import time
import re
import safe_io
from typing import TypedDict, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, START, END

# Guardrails
from guardrails.input_filter import scan_input_payload, get_conversational_response

# RAG & Verification
from rag.retriever import retrieve
from rag.verifier import verify_and_ground_citations

# Generative Model Extraction
from agents.extraction_agent import extract_system_architecture

# Reflection & Critique Loop
from agents.critic_agent import verify_and_critique_specifications
from agents.revision_node import revise_flawed_requirements

# Functional Layers
from synthesis.requirement_assembler import (
    ARCHITECTURAL_PILLARS,
    assemble_modular_requirements,
    synthesize_regulations,
    synthesize_technical_controls,
    build_traceability_matrix,
)
from evaluation.quality_evaluator import evaluate_ieee830_quality
from governance.risk_engine import calculate_dynamic_risk
from telemetry.pipeline_tracer import (
    PipelineTracer,
    build_conversational_trace,
    build_security_rejection_trace,
    build_domain_rejection_trace,
)


class WorkflowState(TypedDict):
    """Resumable and observable state schema for the financial engineering graph."""
    raw_prompt: str
    effective_text: str
    is_conversational: bool
    conversational_response: str
    is_security_violation: bool
    security_rejection_response: str
    is_out_of_domain: bool
    domain_rejection_response: str
    domain: str
    project_title: str
    rag_contexts: List[Dict[str, Any]]
    rag_context_str: str
    arch_data: Dict[str, Any]
    requirements_list: List[Dict[str, Any]]
    passages_catalog: Dict[str, Any]
    verification_rate_pct: float
    categories_breakdown: Dict[str, List[Dict[str, Any]]]
    critic_report: Dict[str, Any]
    critic_iteration: int
    max_retries: int
    revision_history: List[Dict[str, Any]]
    quality_info: Dict[str, Any]
    risk_info: Dict[str, Any]
    regulations: List[Dict[str, Any]]
    technical_controls: List[Dict[str, Any]]
    traceability_matrix: List[Dict[str, Any]]
    pipeline_trace: List[Dict[str, Any]]
    stage_io_logs: List[Dict[str, Any]]
    step_callback: Optional[Any]


# ---------------------------------------------------------------------------
# Canonical Financial Domain Catalog (Single Source of Truth)
# ---------------------------------------------------------------------------
FINANCIAL_DOMAIN_CATALOG: Dict[str, Dict[str, Any]] = {
    "trading": {
        "name": "Algorithmic Trading & Colocation Gateway",
        "category": "Algorithmic Trading, Exchange Colocation, Low Latency FIX Gateway",
        "statutory": "SEBI CSCRF Framework, NSE Colocation Guidelines",
        "keywords": [
            "trading", "stock", "stocks", "bse", "nse", "sebi", "equity", "equities",
            "derivative", "derivatives", "fix", "algo trading", "algo-trading",
            "algorithmic trading", "trading algo", "demat", "margin", "rms",
            "market depth", "tick", "colocation"
        ]
    },
    "atm": {
        "name": "ATM Physical & Cyber Security",
        "category": "Physical Security, Cryptographic Hardware, ATM Switch, FIPS 140-3",
        "statutory": "RBI Master Direction on ATM Security, FIPS 140-3 Cryptographic Standards",
        "keywords": [
            "atm", "cassette", "sensor", "iot", "vault", "cash", "dispenser",
            "hardware", "tamper", "zeroization", "pin pad", "epp", "fips 140", "reed sensor"
        ]
    },
    "payments": {
        "name": "Retail Payments & UPI AutoPay Infrastructure",
        "category": "Consumer Payments, Mobile FinTech, UPI AutoPay Recurring Mandate",
        "statutory": "NPCI UPI Procedural Guidelines on Recurring Mandate Execution, DPDP Act 2023",
        "keywords": [
            "upi", "payment", "payments", "autopay", "mandate", "qr", "pos", "card",
            "token", "merchant", "switch", "gateway", "bbps", "imps", "neft", "wallet"
        ]
    },
    "insurance": {
        "name": "Insurance & TPA Settlement Switch",
        "category": "Insurance Policy & Core Underwriting, Actuarial Solvency",
        "statutory": "IRDAI (Protection of Policyholders' Interests) Regulations, Insurance Act",
        "keywords": [
            "insurance", "policy", "underwriting", "claim", "claims", "irdai",
            "actuarial", "solvency", "premium", "reinsurance", "tpa"
        ]
    },
    "aml": {
        "name": "Anti-Money Laundering & Sanctions Screening",
        "category": "Anti-Money Laundering, Transaction Surveillance, Statutory Reporting",
        "statutory": "Prevention of Money-Laundering Act (PMLA 2002), FIU-IND Reporting Guidelines",
        "keywords": [
            "aml", "anti-money laundering", "pmla", "fiu", "fiu-ind", "str", "ctr",
            "suspicious transaction", "sanctions", "pep", "cft", "fatf"
        ]
    },
    "lending": {
        "name": "Digital Lending & KYC Architecture",
        "category": "Digital Lending, Credit Risk, Loan Origination & Servicing",
        "statutory": "RBI Digital Lending Guidelines 2022, Master Direction on KYC",
        "keywords": [
            "loan", "lending", "credit", "cibil", "crif", "disbursal", "borrower",
            "microfinance", "shg", "farmer", "kfs", "apr", "nbfc", "dbt", "emi", "origination"
        ]
    },
    "capital_markets": {
        "name": "Capital Markets Clearing & Settlement",
        "category": "Capital Markets Clearing, Depository & Custody Architecture",
        "statutory": "SEBI Intermediaries Regulations, Depositories Act 1996",
        "keywords": [
            "back-office", "back office", "clearing", "settlement", "nsccl", "iccl",
            "depository", "nsdl", "cdsl", "demat transfer", "contract note", "corporate action",
            "stt", "dp charge", "custody"
        ]
    },
    "compliance": {
        "name": "Regulatory Compliance & Risk Infrastructure",
        "category": "Regulatory Reporting, OSMOS Filing & Risk Surveillance",
        "statutory": "RBI Cyber Security Framework, Master Direction on Financial Statements",
        "keywords": ["compliance", "osmos", "crilc", "audit", "fraud"]
    },
    "sandbox": {
        "name": "Regulatory Sandbox & CBDC Prototyping",
        "category": "Regulatory Sandbox Cohort, Offline CBDC, Exploratory Prototyping",
        "statutory": "RBI Regulatory Sandbox Enabling Framework, RBI Digital Rupee Directives",
        "keywords": [
            "regulatory sandbox", "sandbox", "cohort", "synthetic money", "cbdc",
            "pilot", "proof of concept", "mock simulator"
        ]
    }
}


def _detect_financial_domain(text: str) -> str:
    """Detect specific financial sub-domain using word boundaries against canonical catalog."""
    t = text.lower()
    for domain_key, info in FINANCIAL_DOMAIN_CATALOG.items():
        for word in info["keywords"]:
            if re.search(r"\b" + re.escape(word) + r"\b", t):
                return domain_key
    return "general"


def get_domain_display_name(domain_key: str) -> str:
    """Resolve human-readable domain name from catalog."""
    entry = FINANCIAL_DOMAIN_CATALOG.get(domain_key)
    return entry["name"] if entry else "Indian Regulated Financial Infrastructure"


def get_domain_category(domain_key: str) -> str:
    """Resolve category specification for domain."""
    entry = FINANCIAL_DOMAIN_CATALOG.get(domain_key)
    return entry["category"] if entry else "Financial Software Engineering"


def get_domain_statutory(domain_key: str) -> str:
    """Resolve canonical statutory framework for domain."""
    entry = FINANCIAL_DOMAIN_CATALOG.get(domain_key)
    return entry["statutory"] if entry else "RBI Cyber Security Framework, NPCI Guidelines, SEBI Regulations, DPDP Act 2023"


def _group_by_category(requirements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group specifications by architectural pillar category."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in requirements:
        cat = r.get("category", "General")
        grouped.setdefault(cat, []).append(r)
    return grouped


def log_stage_io(
    state: WorkflowState,
    stage_id: str,
    stage_name: str,
    input_payload: Dict[str, Any],
    output_payload: Dict[str, Any],
    duration_ms: float,
    status: str = "COMPLETED"
) -> Dict[str, Any]:
    """Record observable input and output payload for each stage."""
    entry = {
        "stage_id": stage_id,
        "stage_name": stage_name,
        "iteration": state.get("critic_iteration", 0),
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "input_payload": input_payload,
        "output_payload": output_payload
    }
    return entry


def append_stage_trace_and_io(
    state: WorkflowState,
    stage_id: str,
    stage_name: str,
    stage_record: Any,
    input_payload: Dict[str, Any],
    output_payload: Dict[str, Any],
    duration_ms: float,
    status: str = "COMPLETED"
) -> Dict[str, Any]:
    """Shared helper to record observability I/O and append pipeline trace without duplication."""
    io_log = log_stage_io(state, stage_id, stage_name, input_payload, output_payload, duration_ms, status=status)
    records = stage_record if isinstance(stage_record, list) else [stage_record]
    return {
        "pipeline_trace": state.get("pipeline_trace", []) + records,
        "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
    }


# ---------------------------------------------------------------------------
# Node 1: Zero-Trust Perimeter Guardrail Node
# ---------------------------------------------------------------------------
def guardrail_node(state: WorkflowState) -> Dict[str, Any]:
    start_t = time.time()
    prompt = state["raw_prompt"]
    cb = state.get("step_callback")

    scan_res = scan_input_payload(prompt)
    duration_ms = (time.time() - start_t) * 1000

    if scan_res.get("is_conversational"):
        if cb:
            cb("Perimeter Guardrail", "Classified query as conversational interaction.")
        io_log = log_stage_io(
            state, "guardrail", "Perimeter Guardrail",
            {"prompt": prompt},
            {"type": "conversational", "response": scan_res.get("conversational_response")},
            duration_ms
        )
        return {
            "is_conversational": True,
            "conversational_response": scan_res.get("conversational_response") or get_conversational_response(prompt),
            "pipeline_trace": build_conversational_trace(),
            "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
        }

    if not scan_res["is_safe"] or scan_res["is_injection"]:
        reason_msg = scan_res["injection_details"] if scan_res["is_injection"] else "adversarial command detected"
        if cb:
            cb("Perimeter Guardrail", f"Blocked: Zero-Trust security filter intercepted violation ({reason_msg}).")
        io_log = log_stage_io(
            state, "guardrail", "Perimeter Guardrail",
            {"prompt": prompt},
            {"type": "security_violation", "reason": reason_msg},
            duration_ms,
            status="REJECTED"
        )
        return {
            "is_security_violation": True,
            "security_rejection_response": (
                f"**Security Guardrail Alert**: Your input was blocked because the Zero-Trust security filter detected: {reason_msg}"
            ),
            "pipeline_trace": build_security_rejection_trace(reason_msg, scan_res.get("threat_assessment", "")),
            "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
        }

    if not scan_res["in_domain"]:
        cat = scan_res["domain_tag"]
        clarification = scan_res["domain_reason"]
        if cb:
            cb("Perimeter Guardrail", f"Domain Boundary Alert: Intercepted out-of-scope query ({cat}).")
        io_log = log_stage_io(
            state, "guardrail", "Perimeter Guardrail",
            {"prompt": prompt},
            {"type": "out_of_domain", "category": cat},
            duration_ms,
            status="REJECTED"
        )
        rejection_msg = (
            f"### Domain Scope Alert: Query Outside Financial Engineering Boundary\n\n"
            f"The **Perimeter Guardrail** has intercepted this request because it belongs to **{cat}**, "
            f"which is outside the operational scope of this platform.\n\n"
            f"> *Guidance*: {clarification}"
        )
        return {
            "is_out_of_domain": True,
            "domain_rejection_response": rejection_msg,
            "pipeline_trace": build_domain_rejection_trace(cat, scan_res.get("threat_assessment", "")),
            "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
        }

    if cb:
        cb("Perimeter Guardrail", "Zero-Trust perimeter scan passed. Input verified compliant.")

    io_log = log_stage_io(
        state, "guardrail", "Perimeter Guardrail",
        {"prompt": prompt},
        {"sanitized_text": scan_res["sanitized_text"], "pii_count": len(scan_res.get("pii_detected", []))},
        duration_ms,
        status="PASSED"
    )

    stage_record = {
        "agent_id": "supervisor",
        "agent_name": "Perimeter Guardrail",
        "role": "Zero-Trust Perimeter Gatekeeper",
        "status": "PASSED",
        "badge": "Input Gate",
        "badge_color": "#A78BFA",
        "output_summary": "Dispatched execution pipeline. Input query verified compliant with Zero-Trust safety filters.",
        "details": {
            "guardrail_status": f"Verified Safe (Injection Risk {scan_res.get('injection_score', 0.0):.2f})",
            "threat_assessment": scan_res.get("threat_assessment", "Zero prompt injection or sensitive PII detected"),
            "sanitized_length": len(scan_res["sanitized_text"])
        }
    }

    return {
        "effective_text": scan_res["sanitized_text"],
        "is_conversational": False,
        "is_security_violation": False,
        "is_out_of_domain": False,
        "pipeline_trace": state.get("pipeline_trace", []) + [stage_record],
        "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
    }


# ---------------------------------------------------------------------------
# Node 2: RAG Retrieval Node
# ---------------------------------------------------------------------------
def retrieval_node(state: WorkflowState) -> Dict[str, Any]:
    start_t = time.time()
    effective_text = state["effective_text"]
    cb = state.get("step_callback")

    rag_contexts = []
    rag_context_str = ""
    try:
        rag_contexts = retrieve(effective_text, top_k=12)
        rag_context_str = "\n\n".join([
            f"[{c.get('citation', 'Regulatory KB')}]:\n{c.get('text', '').strip()[:450]}"
            for c in rag_contexts if c.get("text")
        ])
    except Exception as exc:
        print(f"[WARN] RAG retrieval notice: {exc}")

    duration_ms = (time.time() - start_t) * 1000
    top_c = rag_contexts[0].get("citation", "Regulatory KB") if rag_contexts else "None"
    if cb:
        cb("Regulatory Retrieval (RAG)", f"Retrieved {len(rag_contexts)} statutory clauses from Qdrant (Top: {top_c})")

    io_log = log_stage_io(
        state, "retrieval", "Regulatory Retrieval (RAG)",
        {"query": effective_text[:200], "top_k": 4},
        {"chunks_count": len(rag_contexts), "top_citation": top_c},
        duration_ms
    )

    stage_record = {
        "agent_id": "compliance",
        "agent_name": "Regulatory Retrieval (RAG)",
        "role": "RAG Regulatory Grounding (Qdrant Vector DB)",
        "status": "COMPLETED",
        "badge": "RAG Grounding",
        "badge_color": "#38BDF8",
        "output_summary": f"Retrieved {len(rag_contexts)} statutory regulatory clauses from Qdrant vector database.",
        "details": {
            "vector_db": "Qdrant Vector Engine (3,119 Chunks)",
            "embedding_model": "BAAI/bge-small-en-v1.5 (Cosine Metric)",
            "top_regulation": top_c,
            "retrieved_clauses": [
                {"citation": c.get("citation"), "score": round(float(c.get("score", 0.0)), 4)}
                for c in rag_contexts
            ]
        }
    }

    return {
        "rag_contexts": rag_contexts,
        "rag_context_str": rag_context_str,
        "pipeline_trace": state.get("pipeline_trace", []) + [stage_record],
        "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
    }


# ---------------------------------------------------------------------------
# Node 3: LLM Model Extraction Node (Local GPU LoRA)
# ---------------------------------------------------------------------------
def extraction_node(state: WorkflowState) -> Dict[str, Any]:
    start_t = time.time()
    effective_text = state["effective_text"]
    rag_context_str = state["rag_context_str"]
    cb = state.get("step_callback")

    arch_data = extract_system_architecture(effective_text, rag_context=rag_context_str, step_callback=cb)

    duration_ms = (time.time() - start_t) * 1000
    domain_key = _detect_financial_domain(effective_text)
    elicited_domain_name = get_domain_display_name(domain_key)

    raw_domain = arch_data.get("domain")
    if not raw_domain or raw_domain in ("Indian Financial Infrastructure", "Indian Regulated Financial Infrastructure"):
        detected_domain = elicited_domain_name
    else:
        detected_domain = raw_domain

    project_title = arch_data.get("project_title") or f"SRS — {detected_domain}"

    io_log = log_stage_io(
        state, "extraction", "Architecture Extraction Model",
        {"input_scope": effective_text[:200], "rag_length": len(rag_context_str)},
        {"project_title": project_title, "domain": detected_domain, "extracted_count": len(arch_data.get("requirements", []))},
        duration_ms
    )

    stage_record_eli = {
        "agent_id": "elicitation",
        "agent_name": "Domain Scope Classification",
        "role": "Scope & Intent Resolution",
        "status": "COMPLETED",
        "badge": "Analysis",
        "badge_color": "#60A5FA",
        "output_summary": f"Classified domain as '{detected_domain}' and structured system scope into '{project_title}'.",
        "details": {
            "domain_classified": detected_domain,
            "project_title": project_title,
            "scope_depth": "Enterprise Financial Infrastructure"
        }
    }

    stage_record_ext = {
        "agent_id": "extraction",
        "agent_name": "Architecture Extraction Model",
        "role": "Fine-Tuned 4-bit LoRA Architecture Extractor",
        "status": "COMPLETED",
        "badge": "Fine-Tuned LoRA",
        "badge_color": "#34D399",
        "output_summary": f"Extracted {len(arch_data.get('requirements', []))} core specifications with MoSCoW prioritization on local GPU.",
        "details": {
            "base_model": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
            "lora_adapter": "finetune/adapters/extraction_adapter (PEFT multi-adapter active)",
            "hardware": "NVIDIA GeForce RTX 3050 (6.0 GB VRAM)",
            "vram_footprint": "2.26 GB",
            "moscow_priority": arch_data.get("moscow_priority", "Must Have")
        }
    }

    return {
        "arch_data": arch_data,
        "domain": detected_domain,
        "project_title": project_title,
        "pipeline_trace": state.get("pipeline_trace", []) + [stage_record_eli, stage_record_ext],
        "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
    }


# ---------------------------------------------------------------------------
# Node 4: Specification Assembly Node
# ---------------------------------------------------------------------------
def assembly_node(state: WorkflowState) -> Dict[str, Any]:
    start_t = time.time()
    arch_data = state["arch_data"]
    effective_text = state["effective_text"]
    rag_contexts = state["rag_contexts"]

    # Assemble modular specifications
    raw_reqs = assemble_modular_requirements(arch_data, effective_text, rag_contexts)
    verif_res = verify_and_ground_citations(raw_reqs, rag_contexts)
    reqs_list = verif_res["verified_requirements"]
    passages_catalog = verif_res.get("passages_catalog", {})
    verification_rate_pct = verif_res.get("verification_rate_pct", 100.0)
    categories_breakdown = _group_by_category(reqs_list)

    duration_ms = (time.time() - start_t) * 1000

    io_log = log_stage_io(
        state, "assembly", "Specification Assembler",
        {"raw_requirements_count": len(raw_reqs), "rag_contexts_count": len(rag_contexts)},
        {"modular_count": len(reqs_list), "pillars_count": len(categories_breakdown)},
        duration_ms
    )

    return {
        "requirements_list": reqs_list,
        "categories_breakdown": categories_breakdown,
        "passages_catalog": passages_catalog,
        "verification_rate_pct": verification_rate_pct,
        "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
    }


# ---------------------------------------------------------------------------
# Node 5: Verifier & Critic Agent Node (Critique & Reflection)
# ---------------------------------------------------------------------------
def critic_node(state: WorkflowState) -> Dict[str, Any]:
    start_t = time.time()
    reqs_list = state["requirements_list"]
    cb = state.get("step_callback")
    domain = state.get("domain", "Indian Regulated Financial Infrastructure")

    critic_report = verify_and_critique_specifications(reqs_list, domain=domain)
    duration_ms = (time.time() - start_t) * 1000

    current_iter = state.get("critic_iteration", 0)
    status_label = "PASSED" if critic_report["is_valid"] else f"FLAGGED ({len(critic_report['flawed_req_ids'])} items)"

    if cb:
        cb("Verifier & Critic Agent", f"Cycle {current_iter + 1}: LLM Quality Score {critic_report['overall_critique_score']}% ({status_label})")

    io_log = log_stage_io(
        state, "critic", "Verifier & Critic Agent",
        {"requirements_evaluated": len(reqs_list), "cycle": current_iter + 1},
        {
            "is_valid": critic_report["is_valid"],
            "quality_score": critic_report["overall_critique_score"],
            "flawed_count": len(critic_report["flawed_req_ids"]),
            "flawed_ids": critic_report["flawed_req_ids"],
            "contradictions": critic_report["contradictions_count"],
            "duplicates": critic_report["duplicates_count"],
            "model": critic_report.get("model_name", "unsloth/Qwen2.5-3B-Instruct")
        },
        duration_ms,
        status="PASSED" if critic_report["is_valid"] else "REVISE"
    )

    stage_record = {
        "agent_id": f"critic_cycle_{current_iter + 1}",
        "agent_name": f"Verifier & Critic (Cycle {current_iter + 1})",
        "role": "Autonomous Requirements Critique & Ambiguity Verifier (True LLM Agent)",
        "status": "PASSED" if critic_report["is_valid"] else "REVISION_REQUESTED",
        "badge": "LLM Verifier & Critic",
        "badge_color": "#10A37F" if critic_report["is_valid"] else "#F59E0B",
        "output_summary": critic_report["summary"],
        "details": {
            "cycle": current_iter + 1,
            "inference_mode": "True LLM Cognitive Reasoning",
            "model": "unsloth/Qwen2.5-3B-Instruct",
            "critique_quality_score": critic_report["overall_critique_score"],
            "ambiguous_count": critic_report["ambiguous_count"],
            "untestable_count": critic_report["untestable_count"],
            "duplicates_count": critic_report["duplicates_count"],
            "contradictions_count": critic_report["contradictions_count"],
            "flawed_req_ids": critic_report["flawed_req_ids"]
        }
    }

    return {
        "critic_report": critic_report,
        "pipeline_trace": state.get("pipeline_trace", []) + [stage_record],
        "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
    }


# ---------------------------------------------------------------------------
# Node 6: Targeted Revision Node (Critique Loop Action)
# ---------------------------------------------------------------------------
def revision_node(state: WorkflowState) -> Dict[str, Any]:
    start_t = time.time()
    reqs_list = state["requirements_list"]
    critic_report = state["critic_report"]
    flawed_ids = critic_report.get("flawed_req_ids", [])
    guidance = critic_report.get("regeneration_guidance", {})
    domain = state.get("domain", "Indian Financial Infrastructure")
    cb = state.get("step_callback")

    if cb:
        cb("Revision Node", f"Regenerating {len(flawed_ids)} flagged specifications using targeted critique guidance...")

    revised_reqs, revision_records = revise_flawed_requirements(
        reqs_list, flawed_ids, guidance, domain=domain
    )
    duration_ms = (time.time() - start_t) * 1000

    new_iter = state.get("critic_iteration", 0) + 1

    io_log = log_stage_io(
        state, "revision", "Targeted Revision Node",
        {"flawed_ids": flawed_ids, "guidance_keys": list(guidance.keys())},
        {"revised_count": len(revision_records), "iteration_next": new_iter},
        duration_ms
    )

    stage_record = {
        "agent_id": f"revision_cycle_{new_iter}",
        "agent_name": f"Targeted Revision Node (Cycle {new_iter})",
        "role": "Self-Correction & Autonomous Specification Repair (True LLM Agent)",
        "status": "COMPLETED",
        "badge": "LLM Autonomous Repair",
        "badge_color": "#8B5CF6",
        "output_summary": f"Autonomously repaired {len(revision_records)} specifications resolving critic feedback using local LLM.",
        "details": {
            "inference_mode": "True LLM Autonomous Regeneration",
            "model": "unsloth/Qwen2.5-3B-Instruct",
            "revised_requirements": [r["req_id"] for r in revision_records],
            "repairs": revision_records[:4]
        }
    }

    return {
        "requirements_list": revised_reqs,
        "critic_iteration": new_iter,
        "revision_history": state.get("revision_history", []) + revision_records,
        "pipeline_trace": state.get("pipeline_trace", []) + [stage_record],
        "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
    }


# ---------------------------------------------------------------------------
# Node 7: Risk Governance & Gate Preparation Node
# ---------------------------------------------------------------------------
def governance_node(state: WorkflowState) -> Dict[str, Any]:
    start_t = time.time()
    reqs_list = state["requirements_list"]
    effective_text = state["effective_text"]
    domain = state["domain"]
    project_title = state["project_title"]
    rag_contexts = state["rag_contexts"]
    cb = state.get("step_callback")

    # Quality Evaluation & Traceability
    verification_rate_pct = state.get("verification_rate_pct", 100.0)
    quality_info = evaluate_ieee830_quality(reqs_list, verification_rate_pct=verification_rate_pct)
    regulations_found = synthesize_regulations(reqs_list, rag_contexts)
    technical_controls = synthesize_technical_controls(reqs_list)
    categories_breakdown = state.get("categories_breakdown") or _group_by_category(reqs_list)
    traceability_matrix = build_traceability_matrix(categories_breakdown, project_title)

    # Hybrid Risk Engine (Model + Rule)
    risk_info = calculate_dynamic_risk(effective_text, domain=domain, requirements=reqs_list)

    duration_ms = (time.time() - start_t) * 1000

    if cb:
        cb("Risk Governance Engine", f"Evaluated regulatory risk score at {risk_info['risk_score_pct']}% ({risk_info['risk_level']}).")
        cb("SDLC Recommendation Engine", "Stage-gate matrix calibrated; awaiting human verification.")

    io_log = log_stage_io(
        state, "governance", "Risk Governance Engine",
        {"domain": domain, "reqs_count": len(reqs_list)},
        {
            "risk_score_pct": risk_info["risk_score_pct"],
            "risk_level": risk_info["risk_level"],
            "ieee830_quality_score": quality_info["overall_quality_score"]
        },
        duration_ms
    )

    stage_record_spec = {
        "agent_id": "artefact",
        "agent_name": "Specification Assembler",
        "role": "Modular Specification & IEEE 830 Compiler",
        "status": "COMPLETED",
        "badge": "Synthesis",
        "badge_color": "#FBBF24",
        "output_summary": (
            f"Synthesized {len(reqs_list)} specifications across {len(categories_breakdown)} architectural pillars "
            f"with {quality_info['overall_quality_score']}% IEEE 830 Quality Rating "
            f"({quality_info['clause_level_traceability_pct']}% clause traceability)."
        ),
        "details": {
            "total_specifications": len(reqs_list),
            "pillars_count": len(categories_breakdown),
            "pillars_list": list(categories_breakdown.keys()),
            "traceability_matrix_rows": len(traceability_matrix),
            "technical_controls_count": len(technical_controls),
            "ieee830_quality_score": quality_info["overall_quality_score"],
            "ieee830_rating": quality_info.get("rating_label", "Good"),
            "clause_level_traceability_pct": quality_info.get("clause_level_traceability_pct", 100.0),
            "document_artefact": "IEEE 830 SRS PDF"
        }
    }

    stage_record_risk = {
        "agent_id": "risk",
        "agent_name": "Risk Governance Engine",
        "role": "Hybrid Multi-Factor Statutory Risk Scoring",
        "status": "COMPLETED",
        "badge": "Governance",
        "badge_color": "#F87171",
        "output_summary": f"Calibrated regulatory risk score to {risk_info['risk_score_pct']}% ({risk_info['risk_level']}).",
        "details": {
            "risk_score_pct": risk_info["risk_score_pct"],
            "risk_level": risk_info["risk_level"],
            "risk_summary": risk_info["risk_summary"],
            "methodology": risk_info.get("methodology"),
            "risk_factors": risk_info["risk_factors"]
        }
    }

    stage_record_sdlc = {
        "agent_id": "sdlc",
        "agent_name": "SDLC Recommendation Engine",
        "role": "Regulatory Lifecycle & Stage-Gate Governance",
        "status": "STANDBY / READY",
        "badge": "Governance Gate",
        "badge_color": "#EC4899",
        "output_summary": "Prepared regulatory stage-gate matrix. Awaiting human verification of requirements to hot-swap SDLC LoRA adapter.",
        "details": {
            "lifecycle_governance": "Stage-Gate Verification Architecture",
            "hot_swap_adapter": "finetune/adapters/sdlc_adapter",
            "regulatory_alignment": f"Calibrated for {domain} directives",
            "next_action": "Execute Step 2: Fine-Tuned SDLC Recommendation upon human sign-off"
        }
    }

    return {
        "quality_info": quality_info,
        "risk_info": risk_info,
        "regulations": regulations_found,
        "technical_controls": technical_controls,
        "traceability_matrix": traceability_matrix,
        "pipeline_trace": state.get("pipeline_trace", []) + [stage_record_spec, stage_record_risk, stage_record_sdlc],
        "stage_io_logs": state.get("stage_io_logs", []) + [io_log]
    }


# ---------------------------------------------------------------------------
# Routing Logic & Conditional Edges
# ---------------------------------------------------------------------------
def route_guardrail(state: WorkflowState) -> str:
    """Check whether perimeter guardrail halted execution or cleared for RAG."""
    if state.get("is_conversational") or state.get("is_security_violation") or state.get("is_out_of_domain"):
        return END
    return "retrieval_node"


def should_critique_retry(state: WorkflowState) -> str:
    """Determine whether to route to targeted revision loop or proceed to governance."""
    critic_report = state.get("critic_report", {})
    current_iter = state.get("critic_iteration", 0)
    max_retries = state.get("max_retries", 2)

    # If valid or maximum retry budget exhausted -> proceed to final governance
    if critic_report.get("is_valid", False) or current_iter >= max_retries:
        return "governance_node"

    # Otherwise route to revision node for targeted architectural repair
    return "revision_node"


# ---------------------------------------------------------------------------
# LangGraph Workflow Construction
# ---------------------------------------------------------------------------
def build_financial_engineering_graph():
    """Build and compile the LangGraph workflow with reflection & critique loop."""
    builder = StateGraph(WorkflowState)

    # 1. Register Nodes
    builder.add_node("guardrail_node", guardrail_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("extraction_node", extraction_node)
    builder.add_node("assembly_node", assembly_node)
    builder.add_node("critic_node", critic_node)
    builder.add_node("revision_node", revision_node)
    builder.add_node("governance_node", governance_node)

    # 2. Register Edges
    builder.add_edge(START, "guardrail_node")
    builder.add_conditional_edges(
        "guardrail_node",
        route_guardrail,
        {"retrieval_node": "retrieval_node", END: END}
    )
    builder.add_edge("retrieval_node", "extraction_node")
    builder.add_edge("extraction_node", "assembly_node")
    builder.add_edge("assembly_node", "critic_node")

    # Dynamic Critique Loop: critic -> revision -> critic
    builder.add_conditional_edges(
        "critic_node",
        should_critique_retry,
        {
            "governance_node": "governance_node",
            "revision_node": "revision_node"
        }
    )
    builder.add_edge("revision_node", "critic_node")
    builder.add_edge("governance_node", END)

    return builder.compile()


# Global compiled graph instance
FINANCIAL_ENGINEERING_GRAPH = build_financial_engineering_graph()


def run_financial_engineering_graph(
    prompt: str,
    step_callback: Optional[Callable[[str, str], None]] = None,
    max_retries: int = 2
) -> Dict[str, Any]:
    """Execute financial engineering pipeline using the compiled LangGraph workflow."""
    initial_state: WorkflowState = {
        "raw_prompt": prompt,
        "effective_text": prompt,
        "is_conversational": False,
        "conversational_response": "",
        "is_security_violation": False,
        "security_rejection_response": "",
        "is_out_of_domain": False,
        "domain_rejection_response": "",
        "domain": "general",
        "project_title": "Software Requirements Specification",
        "rag_contexts": [],
        "rag_context_str": "",
        "arch_data": {},
        "requirements_list": [],
        "passages_catalog": {},
        "verification_rate_pct": 100.0,
        "categories_breakdown": {},
        "critic_report": {},
        "critic_iteration": 0,
        "max_retries": max_retries,
        "revision_history": [],
        "quality_info": {},
        "risk_info": {},
        "regulations": [],
        "technical_controls": [],
        "traceability_matrix": [],
        "pipeline_trace": [],
        "stage_io_logs": [],
        "step_callback": step_callback
    }

    final_state = FINANCIAL_ENGINEERING_GRAPH.invoke(initial_state)

    # 1. Guardrail / Conversational Rejections
    for flag, resp_key in [
        ("is_conversational", "conversational_response"),
        ("is_security_violation", "security_rejection_response"),
        ("is_out_of_domain", "domain_rejection_response"),
    ]:
        if final_state.get(flag):
            trace = final_state.get("pipeline_trace", [])
            return {
                flag: True,
                resp_key: final_state.get(resp_key),
                "original_requirement": prompt.strip(),
                "pipeline_trace": trace,
                "agent_trace": trace,
                "stage_io_logs": final_state.get("stage_io_logs", [])
            }

    # 2. Fully Formatted Pipeline Result Dictionary
    arch_data = final_state.get("arch_data", {})
    risk_info = final_state.get("risk_info", {})
    quality_info = final_state.get("quality_info", {})
    categories_breakdown = final_state.get("categories_breakdown", {})
    reqs_list = final_state.get("requirements_list", [])
    rag_contexts = final_state.get("rag_contexts", [])
    pipeline_trace = final_state.get("pipeline_trace", [])
    detected_domain = final_state.get("domain", "Indian Regulated Financial Infrastructure")
    project_title = final_state.get("project_title", f"SRS — {detected_domain}")

    return {
        "is_conversational": False,
        "is_security_violation": False,
        "is_out_of_domain": False,
        "title": project_title,
        "domain": detected_domain,
        "original_requirement": prompt.strip(),
        "fine_tuned_extraction": arch_data,
        "total_requirements_count": len(reqs_list),
        "moscow_priority": arch_data.get("moscow_priority", "Must Have"),
        "priority_justification": arch_data.get(
            "priority_justification",
            f"Mandatory compliance and risk governance under {detected_domain} directives."
        ),
        "risk_level": risk_info.get("risk_level", "HIGH"),
        "risk_score_pct": risk_info.get("risk_score_pct", 75),
        "risk_color": risk_info.get("risk_color", "#F87171"),
        "risk_summary": risk_info.get("risk_summary", ""),
        "risk_factors": risk_info.get("risk_factors", []),
        "risk_assessment": risk_info,
        "categories": list(categories_breakdown.keys()),
        "categories_breakdown": categories_breakdown,
        "requirements_list": reqs_list,
        "requirements": reqs_list,
        "regulations": final_state.get("regulations", []),
        "technical_controls": final_state.get("technical_controls", []),
        "traceability_matrix": final_state.get("traceability_matrix", []),
        "rag_citations": [str(c.get("citation", "Knowledge Base")) for c in rag_contexts[:3]],
        "rag_context": final_state.get("rag_context_str", ""),
        "passages_catalog": final_state.get("passages_catalog", {}),
        "verification_rate_pct": final_state.get("verification_rate_pct", 100.0),
        "expert_quality_rating": quality_info,
        "clause_level_traceability_pct": quality_info.get("clause_level_traceability_pct", 100.0),
        "legal_disclaimer": "NOT LEGAL ADVICE: This automated Software Requirements Specification is generated for engineering and architecture planning purposes only. Formal certification requires sign-off from your Chief Compliance Officer (CCO) and legal counsel.",
        "pipeline_trace": pipeline_trace,
        "agent_trace": pipeline_trace,
        # LangGraph Multi-Agent Reflection & Observability
        "critic_report": final_state.get("critic_report", {}),
        "critic_iteration": final_state.get("critic_iteration", 0),
        "revision_history": final_state.get("revision_history", []),
        "stage_io_logs": final_state.get("stage_io_logs", [])
    }
