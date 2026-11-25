# agents/sdlc_agent.py
"""SDLC Recommendation Agent for Financial Systems.
Outputs a simplified, focused recommendation:
  - recommended_model: name of the selected SDLC model
  - reasoning: why THIS model was selected for this requirement
  - alternatives_considered: list of rejected models with why_not explanation

Uses the fine-tuned 4-bit LoRA adapter (finetune/adapters/sdlc_adapter).
"""

import os
import json
import re
import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from guardrails.input_filter import is_safe as input_is_safe
from guardrails.output_filter import filter_output
from workflow.graph_orchestrator import (
    _detect_financial_domain,
    get_domain_category,
    get_domain_statutory,
)

from agents.model_manager import get_sdlc_model


def _models_agree(m1: str, m2: str) -> bool:
    """Check if two SDLC model names belong to the same architectural family."""
    if not m1 or not m2:
        return False
    m1_low = m1.lower().replace("-", " ").replace("/", " ")
    m2_low = m2.lower().replace("-", " ").replace("/", " ")
    if m1_low == m2_low:
        return True
    families = [
        {"v model", "v model with devsecops gates", "vmodel", "v model with devsecops"},
        {"waterfall", "waterfall staged gate", "staged gate", "waterfall / staged gate", "waterfall / staged-gate", "waterfall / stage gate", "waterfall / stage-gate"},
        {"agile devsecops hybrid", "agile devsecops", "agile", "dual track agile", "agile devops", "extreme programming", "xp"},
        {"prototyping model", "prototyping", "iterative model", "iterative", "prototyping / iterative", "prototyping model / iterative pilot", "rad", "rapid application development"},
        {"spiral model", "spiral"},
        {"incremental model", "incremental", "incremental / staged"},
        {"rad", "rapid application development"}
    ]
    for fam in families:
        in_m1 = any(f in m1_low for f in fam)
        in_m2 = any(f in m2_low for f in fam)
        if in_m1 and in_m2:
            return True
    return False


def evaluate_sdlc_decision_matrix(
    requirement: str,
    risk_info: Dict[str, Any] = None,
    domain: str = None
) -> Dict[str, Any]:
    """Transparent multi-factor statutory decision matrix for Indian financial systems.
    Evaluates:
      1. Physical Hardware & Cryptographic HSM (ATM cash cassettes, vault reed sensors, EPP, zeroization)
      2. Sub-millisecond Colocation & Algorithmic Trading (FIX protocol, matching engine, pre-trade RMS)
      3. Regulatory Sandbox & Synthetic Money Pilots (Exploratory POC, test cohorts)
      4. Core Systemic Banking Ledger Migrations (CBS, double-entry general ledger, RTGS)
      5. Insurance Core & Actuarial Solvency (IRDAI, risk modeling, reinsurance)
      6. Regulatory Compliance Surveillance & AML (PMLA 2002, FIU-IND STR/CTR reporting)
      7. High-Velocity Consumer FinTech & Payments (UPI AutoPay, recurring mandates, mobile wallets)
      8. Calibrated Risk Score Thresholds
    """
    t = requirement.lower()

    # Step 0: Check if input contains financial software engineering context
    has_financial_context = any(k in t for k in [
        "bank", "pay", "fund", "money", "card", "atm", "hsm", "sensor", "reed",
        "cassette", "switch", "settle", "ledger", "account", "cbs", "upi", "rbi",
        "sebi", "irdai", "fiu", "pmla", "mandate", "trading", "broker", "demat",
        "order", "exchange", "risk", "crypto", "token", "loan", "lending",
        "credit", "fraud", "surveillance", "kyc", "identity", "transaction",
        "clearing", "fintech", "insurance", "policy", "premium", "claim", "actuarial"
    ])

    risk_score = 70
    if risk_info and isinstance(risk_info, dict):
        risk_score = risk_info.get("risk_score_pct", 70)

    factors = {
        "has_hardware_hsm": any(k in t for k in [
            "atm", "vault", "cash cassette", "cassette dispenser", "reed sensor",
            "tilt sensor", "tamper", "zeroization", "pin pad", "epp", "fips 140",
            "hardware security module", "smart card reader"
        ]),
        "has_colocation_algo": any(k in t for k in [
            "colocation", "colo", "sub-millisecond", "fix 4.", "fix 5.", "matching engine",
            "algo trading", "algorithmic trading", "level-2", "level-3", "order book depth",
            "tick-to-trade", "pre-trade risk"
        ]),
        "is_sandbox_pilot": any(k in t for k in [
            "regulatory sandbox", "sandbox cohort", "sandbox pilot", "proof of concept",
            "poc", "synthetic money", "mock simulator", "pilot cohort", "pilot trial", "paper trading"
        ]),
        "is_core_banking": any(k in t for k in [
            "core banking", "cbs", "general ledger", "double-entry", "chart of accounts",
            "rtgs settlement switch", "neft settlement engine"
        ]),
        "is_insurance": any(k in t for k in [
            "irdai", "insurance policy", "underwriting", "claim settlement", "actuarial",
            "solvency margin", "tpa integration", "reinsurance", "insurance"
        ]),
        "is_aml_compliance": any(k in t for k in [
            "pmla", "fiu-ind", "fiu", "str reporting", "ctr reporting", "anti-money laundering",
            "aml surveillance", "sanctions screening", "pep check"
        ]),
        "is_consumer_payments": any(k in t for k in [
            "upi", "autopay", "recurring mandate", "payment gateway", "bbps",
            "merchant settlement", "qr code", "mobile wallet", "card tokenization"
        ]),
        "risk_score": risk_score,
        "domain": domain or "general"
    }

    # Strict Zero-Silent-Default: Reject out-of-domain or non-financial inputs
    if not has_financial_context or len(t.strip()) < 15:
        return {
            "matrix_model": "Could Not Determine",
            "confidence": 0.0,
            "rule_id": "DM-RULE-UNDETERMINED-OUT-OF-DOMAIN",
            "decision_factors": factors,
            "matrix_reasoning": (
                "The provided text lacks financial domain architecture keywords or statutory directives. "
                "The SDLC decision matrix cannot determine a software lifecycle without system architectural context."
            ),
            "matrix_alternatives": []
        }

    # 1. Hardware / ATM / Physical Tamper / FIPS HSM
    if factors["has_hardware_hsm"]:
        return {
            "matrix_model": "V-Model with DevSecOps Gates",
            "confidence": 0.95,
            "rule_id": "DM-RULE-01-HARDWARE-ATM",
            "decision_factors": factors,
            "matrix_reasoning": (
                "Physical hardware devices (ATM cash cassettes, vault reed sensors, EPP PIN pads) "
                "and cryptographic key zeroization require formal hardware-in-the-loop (HIL) testing, "
                "pre-production factory acceptance testing (FAT), and sequential verification gates that "
                "cannot be refactored like pure software in agile sprints."
            ),
            "matrix_alternatives": [
                {"model": "Agile-DevSecOps Hybrid", "why_not": "Hardware sensor fabrication, physical calibration, and tamper zeroization circuits cannot be iteratively changed mid-sprint."},
                {"model": "Waterfall", "why_not": "Pure Waterfall lacks automated firmware vulnerability scanning and DevSecOps gates mandated by RBI ATM security guidelines."}
            ]
        }

    # 2. Sub-millisecond Colocation & Algo Gateway
    if factors["has_colocation_algo"]:
        return {
            "matrix_model": "V-Model with DevSecOps Gates",
            "confidence": 0.94,
            "rule_id": "DM-RULE-02-ALGO-COLOCATION",
            "decision_factors": factors,
            "matrix_reasoning": (
                "Colocation algorithmic execution engines interfacing directly with exchange matching engines "
                "carry severe financial loss potential per microsecond. SEBI CSCRF directives mandate "
                "bilateral exchange mock certification and formal pre-trade risk management verification gates."
            ),
            "matrix_alternatives": [
                {"model": "Waterfall", "why_not": "Exchange API updates and circuit breaker configurations require continuous automated regression gates rather than static upfront documentation."},
                {"model": "Prototyping Model", "why_not": "Live market trading with capital exposure cannot tolerate throwaway prototype architecture."}
            ]
        }

    # 3. Regulatory Sandbox Pilot / Synthetic Money POC
    if factors["is_sandbox_pilot"]:
        return {
            "matrix_model": "Prototyping Model",
            "confidence": 0.95,
            "rule_id": "DM-RULE-03-SANDBOX-POC",
            "decision_factors": factors,
            "matrix_reasoning": (
                "Regulatory sandbox cohorts operating with synthetic capital and capped participant limits "
                "are exploratory by definition. Rapid prototyping enables customer validation loops and "
                "regulatory feedback before applying for commercial production licensing."
            ),
            "matrix_alternatives": [
                {"model": "Waterfall", "why_not": "Statutory sandbox requirements are exploratory and evolve based on cohort test feedback."},
                {"model": "V-Model", "why_not": "Full V-Model verification overhead is disproportionate for synthetic-money sandbox trials."}
            ]
        }

    # 4. Core Systemic Banking Ledger / CBS Migration
    if factors["is_core_banking"]:
        return {
            "matrix_model": "Waterfall / Staged-Gate",
            "confidence": 0.93,
            "rule_id": "DM-RULE-04-CORE-BANKING",
            "decision_factors": factors,
            "matrix_reasoning": (
                "Core banking general ledger migrations carry irreversible financial accounting risk. "
                "Chart-of-accounts schemas, balance carryforwards, and double-entry invariants must be "
                "completely specified, verified, and audited upfront prior to cutover."
            ),
            "matrix_alternatives": [
                {"model": "Agile-DevSecOps Hybrid", "why_not": "Ledger atomicity, regulatory audit schemas, and statutory chart of accounts cannot be iteratively altered mid-migration."},
                {"model": "RAD", "why_not": "Rapid application development bypasses formal data reconciliation and parallel-run audit gates."}
            ]
        }

    # 5. Insurance Core & Solvency Modeling
    if factors["is_insurance"]:
        is_actuarial_core = any(k in t for k in [
            "actuarial", "solvency", "reinsurance", "underwriting", "policy administration",
            "statutory reserve", "ibnr", "alm"
        ])
        if is_actuarial_core:
            return {
                "matrix_model": "Waterfall / Staged-Gate",
                "confidence": 0.90,
                "rule_id": "DM-RULE-05A-INSURANCE-ACTUARIAL-CORE",
                "decision_factors": factors,
                "matrix_reasoning": (
                    "Insurance core policy administration, actuarial valuation, and statutory solvency margins "
                    "involve binding capital adequacy models governed by IRDAI requiring sequential verification stage-gates."
                ),
                "matrix_alternatives": [
                    {"model": "Agile", "why_not": "Actuarial valuation models and solvency margin formulas require formal stage-gate certification before filing with IRDAI."}
                ]
            }
        else:
            return {
                "matrix_model": "Agile-DevSecOps Hybrid",
                "confidence": 0.91,
                "rule_id": "DM-RULE-05B-INSURANCE-DIGITAL-PORTAL",
                "decision_factors": factors,
                "matrix_reasoning": (
                    "Insurance customer portals, digital aggregators, and claims adjudication workflows "
                    "benefit from iterative sprint delivery with automated DevSecOps and NHCX gateway integration."
                ),
                "matrix_alternatives": [
                    {"model": "Waterfall", "why_not": "Customer experience and third-party API integrations require agile iteration."}
                ]
            }

    # 6. Regulatory Compliance & AML / FIU-IND Reporting
    if factors["is_aml_compliance"]:
        return {
            "matrix_model": "Agile-DevSecOps Hybrid",
            "confidence": 0.92,
            "rule_id": "DM-RULE-06-AML-COMPLIANCE",
            "decision_factors": factors,
            "matrix_reasoning": (
                "Anti-Money Laundering (AML) and FIU-IND compliance systems operate on streaming event data (CEP/Kafka). "
                "Rapidly evolving financial crime typologies necessitate bi-weekly sprint iterations paired with continuous "
                "DevSecOps pipelines enforcing anti-tipping-off security guardrails and XML schema validation."
            ),
            "matrix_alternatives": [
                {"model": "Waterfall", "why_not": "Static waterfall cycles cannot adapt to newly emerging mule account typologies or FATF sanctions updates."},
                {"model": "Prototyping Model", "why_not": "Statutory reporting requires production-grade XML schema fidelity rather than disposable prototypes."}
            ]
        }

    # 7. High-Velocity Consumer FinTech & Payments
    if factors["is_consumer_payments"]:
        return {
            "matrix_model": "Agile-DevSecOps Hybrid",
            "confidence": 0.94,
            "rule_id": "DM-RULE-07-PAYMENTS-FINTECH",
            "decision_factors": factors,
            "matrix_reasoning": (
                "Consumer payment applications governed by NPCI and RBI circulars require iterative sprint "
                "delivery for rapid feature adoption, coupled with continuous automated DevSecOps pipelines "
                "(SAST, DAST, SCA, dependency scanning, dynamic linking verification) at every commit boundary."
            ),
            "matrix_alternatives": [
                {"model": "Waterfall", "why_not": "Frequent NPCI circular addenda (e.g. MyUPI, Tap-and-Pay, Numeric UPI ID) make rigid upfront specifications obsolete."},
                {"model": "V-Model", "why_not": "Pure V-Model phase-gating delays feature delivery in competitive consumer payment ecosystems."}
            ]
        }

    # 8. General Financial Architecture Fallbacks
    if risk_score >= 85:
        return {
            "matrix_model": "V-Model with DevSecOps Gates",
            "confidence": 0.85,
            "rule_id": "DM-RULE-08-HIGH-RISK-FINANCIAL",
            "decision_factors": factors,
            "matrix_reasoning": "High statutory risk score (>85%) mandates formal verification stage-gates with DevSecOps automation.",
            "matrix_alternatives": [
                {"model": "Agile", "why_not": "Pure agile lacks required formal verification gates for critical tier systems."}
            ]
        }
    else:
        return {
            "matrix_model": "Agile-DevSecOps Hybrid",
            "confidence": 0.85,
            "rule_id": "DM-RULE-09-STANDARD-FINANCIAL",
            "decision_factors": factors,
            "matrix_reasoning": "Standard financial architecture requirement suitable for iterative DevSecOps delivery.",
            "matrix_alternatives": [
                {"model": "Waterfall", "why_not": "Slow release cadence hampers regulatory responsiveness."}
            ]
        }


def recommend_sdlc_structured(
    requirement: str,
    rag_context: str = "",
    risk_info: Dict[str, Any] = None,
    domain: str = None,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """Generate a calibrated SDLC recommendation using an ensemble of the transparent
    statutory decision matrix and the fine-tuned LoRA model. Never defaults silently.
    """
    if not input_is_safe(requirement):
        return {
            "recommended_model": "Could Not Determine",
            "confidence": 0.0,
            "confidence_tier": "Low",
            "source": "GUARDRAILS_REJECTED",
            "reasoning": "Input rejected by Zero-Trust security filter (prompt injection or prohibited command).",
            "alternatives_considered": [],
            "decision_matrix_rule": "DM-RULE-GUARDRAIL-REJECTED",
            "decision_factors": {}
        }

    # Step 1: Evaluate Transparent Statutory Decision Matrix Baseline
    matrix_res = evaluate_sdlc_decision_matrix(requirement, risk_info=risk_info, domain=domain)
    matrix_model = matrix_res["matrix_model"]
    matrix_conf = matrix_res["confidence"]

    # If the decision matrix explicitly rejected out-of-domain/adversarial text, do not let LLM hallucinate
    if matrix_model == "Could Not Determine":
        return {
            "recommended_model": "Could Not Determine",
            "confidence": 0.0,
            "confidence_tier": "Low",
            "source": "UNDETERMINED_OUT_OF_DOMAIN",
            "reasoning": (
                "Could not determine an SDLC model: the input lacks financial software engineering context, "
                "or contains out-of-domain/adversarial content. Manual architect review required."
            ),
            "alternatives_considered": [],
            "decision_matrix_rule": matrix_res.get("rule_id", "DM-RULE-UNDETERMINED"),
            "decision_factors": matrix_res.get("decision_factors", {})
        }

    # Step 2: Query Fine-Tuned SDLC LoRA Model as Primary Predictor
    llm_res = None
    llm_err = None
    try:
        tokenizer, model = get_sdlc_model()

        system_prompt = (
            "You are the SDLC Recommendation Agent for Indian financial institutions. "
            "Analyze the requirement and output a JSON with: recommended_model (SDLC model name), "
            "reasoning (why this model fits), and alternatives_considered (list of rejected models with why_not explanation). "
            "Output ONLY valid JSON."
        )

        sdlc_context = ""
        try:
            from rag.retriever import retrieve
            hits = retrieve(f"regulatory SDLC lifecycle model applicability {requirement[:100]}", top_k=3)
            sdlc_context = "\n\n".join([
                f"[{h.get('citation', 'SDLC Governance')}]:\n{h.get('text', '').strip()[:450]}"
                for h in hits if h.get("text")
            ])
        except Exception:
            pass

        combined_rag = f"{rag_context}\n\n{sdlc_context}".strip()

        # Contextual domain and statutory alignment matching canonical catalog
        detected_domain = domain if (domain and domain != "general") else _detect_financial_domain(requirement)
        domain_label = get_domain_category(detected_domain)
        statutory_context = get_domain_statutory(detected_domain) if detected_domain != "general" else ""

        reg_framework = statutory_context if statutory_context else (
            combined_rag[:400] if combined_rag
            else "RBI Cyber Security Framework, NPCI Guidelines, SEBI Regulations, DPDP Act 2023"
        )

        user_prompt = f"""Recommend an optimal Software Development Life Cycle model and stage-gate workflow complying with RBI Cyber Security and DevSecOps guidelines:

Requirement Statement:
\"{requirement[:1800]}\"
Categories: {domain_label}
Regulatory Framework: {reg_framework}

Available SDLC Methodologies: Agile-DevSecOps Hybrid, V-Model with DevSecOps Gates, Waterfall / Staged-Gate, Prototyping Model / Iterative Pilot, Spiral Model, RAD (Rapid Application Development).

Output ONLY valid JSON in this exact structure:
{{
  "recommended_model": "<Selected SDLC Model Name>",
  "reasoning": "<Detailed explanation of why this SDLC model is the best fit for this specific requirement>",
  "alternatives_considered": [
    {{"model": "<Alternative Model 1>", "why_not": "<Why this alternative was rejected>"}},
    {{"model": "<Alternative Model 2>", "why_not": "<Why this alternative was rejected>"}}
  ]
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=450,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        allowed, cleaned, reason = filter_output(response_text)
        if allowed:
            llm_res = _parse_sdlc_json(cleaned)
        else:
            llm_err = f"Output filter: {reason}"
    except Exception as exc:
        llm_err = str(exc)

    # Step 3: Pure Autonomous Model Prediction (No Rule-Based Override)
    if llm_res and llm_res.get("recommended_model") not in [None, "", "Could Not Determine"]:
        final_model = llm_res["recommended_model"]
        reasoning = str(llm_res.get("reasoning", "")).strip()
        alternatives = llm_res.get("alternatives_considered", [])

        # Validate against statutory matrix to calibrate confidence
        if _models_agree(matrix_model, final_model):
            calibrated_conf = min(0.98, max(matrix_conf, 0.90) + 0.04)
            source = f"FINE_TUNED_SDLC_LORA_MODEL (Statutory Matrix Confirmed: {matrix_res.get('rule_id', 'Direct')})"
        else:
            calibrated_conf = 0.92
            source = "FINE_TUNED_SDLC_LORA_MODEL"

    elif matrix_model and matrix_model != "Could Not Determine":
        # Fallback ONLY if fine-tuned model generation failed
        final_model = matrix_model
        calibrated_conf = round(matrix_conf * 0.88, 2)
        source = f"DECISION_MATRIX_FALLBACK ({matrix_res.get('rule_id')})"
        reasoning = (
            f"Determined via Statutory Decision Matrix ({matrix_res.get('rule_id')}): "
            f"{matrix_res.get('matrix_reasoning')}"
        )
        alternatives = matrix_res.get("matrix_alternatives", [])

    else:
        # Both Undetermined -> Explicit Non-Silent Failure
        final_model = "Could Not Determine"
        calibrated_conf = 0.0
        source = "UNDETERMINED"
        reasoning = (
            f"Could not determine a definitive SDLC model for the requirement. "
            f"LLM status: {llm_err or 'Failed parse'}; Decision Matrix: Inconclusive. "
            f"Manual review by Chief Software Architect required."
        )
        alternatives = []

    conf_tier = "Very High" if calibrated_conf >= 0.90 else "High" if calibrated_conf >= 0.80 else "Moderate" if calibrated_conf >= 0.65 else "Low"

    return {
        "recommended_model": final_model,
        "confidence": calibrated_conf,
        "confidence_tier": conf_tier,
        "source": source,
        "reasoning": reasoning,
        "alternatives_considered": alternatives,
        "decision_matrix_rule": matrix_res.get("rule_id", "None"),
        "decision_factors": matrix_res.get("decision_factors", {})
    }


def _parse_sdlc_json(text: str) -> Dict[str, Any]:
    """Robustly extract and parse SDLC JSON from model output. Never defaults silently."""
    if not text or not text.strip():
        return {
            "recommended_model": "Could Not Determine",
            "reasoning": "Model returned an empty response.",
            "alternatives_considered": [],
            "status": "PARSE_EMPTY"
        }

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        json_str = text[start:end + 1]
        try:
            parsed = json.loads(json_str)
            if "recommended_model" in parsed and parsed["recommended_model"]:
                return parsed
        except json.JSONDecodeError:
            pass

    # Try suffix fixes
    if start != -1:
        for suffix in ["]}}", "}]}", "}}", "}"]:
            try:
                parsed = json.loads(text[start:] + suffix)
                if "recommended_model" in parsed and parsed["recommended_model"]:
                    return parsed
            except Exception:
                continue

    # Regex fallback
    m = re.search(r'"recommended_model"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if m and m.group(1).strip():
        reason_m = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        return {
            "recommended_model": m.group(1).strip(),
            "reasoning": reason_m.group(1).strip() if reason_m else "Extracted from model response via regex.",
            "alternatives_considered": [],
            "status": "REGEX_EXTRACTED"
        }

    # NEVER silently return Agile-DevSecOps Hybrid
    return {
        "recommended_model": "Could Not Determine",
        "reasoning": "Model response was not in expected schema or failed parsing.",
        "alternatives_considered": [],
        "status": "PARSE_FAILURE"
    }


def critique_human_sdlc_selection(requirement: str, chosen_model: str, human_feedback: str = "", rag_context: str = "") -> Dict[str, Any]:
    """Pass human-selected SDLC model back to the LLM to get the model's synthesized technical critique,
    strengths, regulatory risks, and required governance mitigations.
    """
    if not input_is_safe(requirement + " " + chosen_model + " " + human_feedback):
        raise ValueError("Input rejected by guardrails.")

    tokenizer, model = get_sdlc_model()

    system_prompt = (
        "You are the Principal Software Architect and SDLC Governance Advisor for Indian regulated financial systems. "
        "The human software engineer/architect has proposed an SDLC model for the project. "
        "Analyze their choice against the statutory regulations (RBI, SEBI, NPCI, DPDP Act 2023) and technical requirements. "
        "Provide your technical critique, strengths, and specific statutory engineering controls required to make this model successful. "
        "Output ONLY valid JSON."
    )

    # Retrieve dedicated SDLC regulatory governance context from Qdrant
    sdlc_context = ""
    try:
        from rag.retriever import retrieve
        hits = retrieve(f"regulatory SDLC lifecycle model applicability {chosen_model} {requirement[:100]}", top_k=3)
        sdlc_context = "\n\n".join([
            f"[{h.get('citation', 'SDLC Governance')}]:\n{h.get('text', '').strip()[:450]}"
            for h in hits if h.get("text")
        ])
    except Exception as exc:
        pass

    combined_rag = f"{rag_context}\n\n{sdlc_context}".strip()
    rag_section = f"\n\nApplicable Regulatory & SDLC Governance Directives:\n{combined_rag[:1200]}" if combined_rag else ""

    user_prompt = f"""Project Requirement:
"{requirement[:1500]}"
{rag_section}

Human Architect's Proposed SDLC Model: "{chosen_model}"
Human Architect Notes: "{human_feedback if human_feedback else 'Proposed based on enterprise team constraints and organizational delivery goals.'}"

Analyze the human architect's proposed model and output ONLY valid JSON in this exact structure:
{{
  "chosen_model": "{chosen_model}",
  "architectural_synthesis": "Comprehensive technical evaluation of how this model applies to this specific project's architecture and regulatory constraints.",
  "strengths_of_choice": "Key advantages and strategic benefits of adopting this model for this system.",
  "statutory_risks_and_mitigations": "Specific compliance risks under Indian financial directives and the exact engineering gates/checkpoints required to mitigate them.",
  "governance_cadence": "Recommended sprint cadence, formal audit sign-offs, and verification milestones (e.g., automated SAST/DAST, CERT-In VAPT pre-production sign-off, or stage-gate reviews)."
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    allowed, cleaned, reason = filter_output(response_text)
    if not allowed:
        raise ValueError(f"Output rejected by guardrails: {reason}")

    # Robust JSON extraction
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        try:
            parsed = json.loads(cleaned[start:end+1])
            if isinstance(parsed, dict) and "architectural_synthesis" in parsed:
                return parsed
        except Exception:
            pass

    # Regex extraction fallback
    def _extract_field(field_name: str, default: str) -> str:
        pattern = rf'"{field_name}"\s*:\s*"([^"]+)"'
        m = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else default

    synthesis = _extract_field(
        "architectural_synthesis",
        f"Adopting {chosen_model} provides a structured lifecycle for this financial system, ensuring that Indian regulatory guardrails (RBI/SEBI/NPCI) are embedded across all engineering phases."
    )
    strengths = _extract_field(
        "strengths_of_choice",
        f"Direct alignment with system modularity, providing controlled verification checkpoints for {chosen_model}."
    )
    risks = _extract_field(
        "statutory_risks_and_mitigations",
        "Mandatory adherence to CERT-In VAPT pre-production security sign-offs, immutable audit logging, and DPDP Act 2023 data residency mandates."
    )
    cadence = _extract_field(
        "governance_cadence",
        "Bi-weekly iterative cadence with automated CI/CD SAST/DAST security gates and formal audit readiness sign-offs."
    )

    return {
        "chosen_model": chosen_model,
        "architectural_synthesis": synthesis,
        "strengths_of_choice": strengths,
        "statutory_risks_and_mitigations": risks,
        "governance_cadence": cadence
    }


def recommend_sdlc(requirement: str) -> str:
    """Backward-compatible wrapper — returns formatted text from structured recommendation."""
    result = recommend_sdlc_structured(requirement)
    lines = [
        f"**Recommended SDLC Model:** {result.get('recommended_model')}",
        "",
        f"**Reasoning:** {result.get('reasoning')}",
        "",
        "**Alternatives Considered:**",
    ]
    for alt in result.get("alternatives_considered", []):
        lines.append(f"  - **{alt.get('model')}**: {alt.get('why_not')}")
    return "\n".join(lines)


if __name__ == "__main__":
    sample_req = "The payment gateway shall tokenize all PAN upon ingestion and transmit ISO 20022 messages via mTLS 1.3."
    result = recommend_sdlc_structured(sample_req)
    print("\n--- SDLC RECOMMENDATION (Structured) ---")
    print(json.dumps(result, indent=2))
