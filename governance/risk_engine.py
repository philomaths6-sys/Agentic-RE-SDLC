# governance/risk_engine.py
"""Hybrid (Model + Rule-Based) Financial Risk Scoring Engine.

Combines:
  1. Statutory Decision Matrix: Deterministic regulatory baselines, exposure weights,
     mitigating factors, and non-negotiable statutory floors/ceilings (RBI/SEBI/NPCI).
  2. Local LLM Semantic Evaluator: Deep architectural risk reasoning and failure vector
     extraction on the local GPU model (Qwen-2.5-3B).
  3. Ensemble Stage-Gate Blending: 60% Statutory Rules + 40% Model Reasoning,
     strictly bounded by statutory safety constraints.
"""

import re
import json
from typing import List, Dict, Any, Optional


def _evaluate_statutory_rule_risk(user_text: str, domain: str) -> Dict[str, Any]:
    """Deterministic statutory risk evaluation against Indian financial regulatory directives."""
    u = user_text.lower()
    factors = []
    factor_breakdown = []

    # 1. Calibrated Domain Baselines
    domain_baselines = {
        "trading": (55, "Algorithmic Capital Markets Execution Baseline"),
        "atm": (55, "Physical Hardware & Cash Dispensing Baseline"),
        "payments": (45, "Consumer Retail Payment Processing Baseline"),
        "lending": (42, "Digital Credit & Loan Origination Baseline"),
        "aml": (38, "Statutory AML / PMLA Monitoring Baseline"),
        "compliance": (35, "Regulatory Compliance & Reporting Baseline"),
        "general": (35, "Regulated Financial Architecture Baseline"),
    }

    base_score, base_rationale = domain_baselines.get(domain.lower(), (35, "Regulated Financial Architecture Baseline"))
    score = base_score
    factor_breakdown.append({
        "category": "Baseline",
        "factor": base_rationale,
        "delta": base_score,
        "impact": f"+{base_score}"
    })

    statutory_floor = None
    statutory_ceiling = None

    # 2. Capital & Financial Execution Exposure
    if any(k in u for k in ["algo", "algorithmic", "high frequency", "colocation", "sub-millisecond", "fix 4", "fix 5", "order book", "matching engine"]):
        score += 24
        statutory_floor = 75  # SEBI CSCRF Tier-1 Floor
        msg = "Sub-millisecond automated order execution and colocation capital exposure (Statutory Floor: 75%)."
        factors.append(msg)
        factor_breakdown.append({"category": "Capital Execution", "factor": msg, "delta": 24, "impact": "+24"})
    elif any(k in u for k in ["core banking", "general ledger", "double-entry", "cbs", "rtgs", "high-value"]):
        score += 20
        msg = "Core banking ledger atomicity with irreversible financial states."
        factors.append(msg)
        factor_breakdown.append({"category": "Capital Execution", "factor": msg, "delta": 20, "impact": "+20"})
    elif any(k in u for k in ["disbursal", "lending", "credit line", "autopay", "recurring debit", "mandate"]):
        score += 14
        msg = "Direct automated fund movement, recurring debit, or credit disbursement facility."
        factors.append(msg)
        factor_breakdown.append({"category": "Capital Execution", "factor": msg, "delta": 14, "impact": "+14"})
    elif any(k in u for k in ["upi", "payment gateway", "switch", "settlement", "merchant", "pos"]):
        score += 10
        msg = "Retail payment transaction routing and merchant settlement."
        factors.append(msg)
        factor_breakdown.append({"category": "Capital Execution", "factor": msg, "delta": 10, "impact": "+10"})

    # 3. Hardware & Physical Security Exposure
    if any(k in u for k in ["atm", "vault", "cash dispenser", "cassette", "sensor", "reed sensor", "tilt", "tamper", "zeroization", "pin pad", "epp"]):
        score += 25
        statutory_floor = 75  # RBI ATM Physical Security Floor
        msg = "Physical hardware tampering, vault sensors, and automated cryptographic key zeroization (Statutory Floor: 75%)."
        factors.append(msg)
        factor_breakdown.append({"category": "Hardware Security", "factor": msg, "delta": 25, "impact": "+25"})

    # 4. PII & Biometric Privacy Exposure
    if any(k in u for k in ["biometric", "facial", "liveness", "v-cip", "aadhaar", "uidai", "dpdp act", "data vault"]):
        score += 14
        msg = "High-volume sensitive citizen biometric data custody or Aadhaar Data Vault under DPDP Act 2023."
        factors.append(msg)
        factor_breakdown.append({"category": "Data Privacy", "factor": msg, "delta": 14, "impact": "+14"})

    # 5. Regulatory Severity & Sanctions
    if any(k in u for k in ["sebi cscrf", "pmla", "fiu-ind", "sanctions", "aml audit"]):
        score += 12
        msg = "Statutory regulatory compliance mandates with criminal liability and mandatory disclosure."
        factors.append(msg)
        factor_breakdown.append({"category": "Regulatory Severity", "factor": msg, "delta": 12, "impact": "+12"})

    # 6. Mitigating Factors & Statutory Ceilings
    if any(k in u for k in ["sandbox", "proof of concept", "poc", "synthetic money", "paper trading", "simulator", "mock"]):
        score -= 32
        statutory_ceiling = 45  # Regulatory Sandbox Statutory Ceiling
        msg = "Controlled regulatory sandbox/simulation environment with synthetic capital (Statutory Ceiling: 45%)."
        factors.append(msg)
        factor_breakdown.append({"category": "Mitigating Factor", "factor": msg, "delta": -32, "impact": "-32"})

    if any(k in u for k in ["read-only", "read only", "reporting only", "internal portal", "view only", "audit viewer", "dashboard"]):
        score -= 26
        statutory_ceiling = min(statutory_ceiling or 40, 35)  # Read-only Statutory Ceiling
        msg = "Non-transactional, read-only analytics/reporting view with zero direct write access (Statutory Ceiling: 35%)."
        factors.append(msg)
        factor_breakdown.append({"category": "Mitigating Factor", "factor": msg, "delta": -26, "impact": "-26"})

    if any(k in u for k in ["pilot trial", "pilot cohort", "capped at", "maximum wallet", "restricted users"]):
        score -= 14
        msg = "Bounded pilot cohort with capped participant volume and restricted monetary balances."
        factors.append(msg)
        factor_breakdown.append({"category": "Mitigating Factor", "factor": msg, "delta": -14, "impact": "-14"})

    # Clamp score
    rule_score = max(18, min(98, score))

    return {
        "rule_score": rule_score,
        "factors": factors,
        "factor_breakdown": factor_breakdown,
        "statutory_floor": statutory_floor,
        "statutory_ceiling": statutory_ceiling
    }


def _evaluate_model_semantic_risk(user_text: str, domain: str) -> Dict[str, Any]:
    """Invoke local GPU model to extract qualitative architectural risk vectors and semantic score."""
    try:
        import torch
        from agents.model_manager import get_extraction_model
        tokenizer, model = get_extraction_model()

        system_prompt = (
            "You are the Chief Risk Officer and Lead Security Architect for Indian regulated financial systems. "
            "Analyze the software requirement scope and evaluate architectural risk. "
            "Output ONLY a valid JSON object with keys: "
            "'model_risk_score' (integer 0-100), "
            "'architectural_risk_level' ('CRITICAL', 'HIGH', 'MEDIUM', 'MODERATE / LOW', or 'MINIMAL'), "
            "'risk_rationale' (concise 1-2 sentence architectural risk explanation), "
            "'identified_risk_vectors' (array of 2-3 specific architectural vulnerability or failure modes). "
            "Output ONLY valid JSON."
        )

        user_prompt = f"""Evaluate architectural risk for this financial software project:
Domain: {domain}
Requirement Scope: "{user_text[:400]}"

JSON Output:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id
            )

        resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        clean_text = resp
        if "```" in clean_text:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
            if m:
                clean_text = m.group(1).strip()
        first_b = clean_text.find("{")
        last_b = clean_text.rfind("}")
        if first_b != -1 and last_b != -1 and last_b > first_b:
            clean_text = clean_text[first_b:last_b+1]

        data = json.loads(clean_text)
        score = int(data.get("model_risk_score", 50))
        return {
            "success": True,
            "model_risk_score": max(10, min(95, score)),
            "model_risk_level": data.get("architectural_risk_level", "MEDIUM"),
            "risk_rationale": data.get("risk_rationale", "Evaluated via local architecture risk model."),
            "identified_risk_vectors": data.get("identified_risk_vectors", [])
        }
    except Exception as exc:
        return {
            "success": False,
            "model_risk_score": None,
            "model_risk_level": None,
            "risk_rationale": f"Statutory rule baseline active (Model semantic pass: {exc})",
            "identified_risk_vectors": []
        }


def calculate_dynamic_risk(
    user_text: str,
    domain: str = "general",
    requirements: Optional[List[Dict[str, Any]]] = None,
    use_model: bool = True
) -> Dict[str, Any]:
    """Execute Hybrid (Model + Rule-Based) Risk Evaluation.
    
    Blends:
      - 60% Statutory Decision Matrix (deterministic regulatory baselines & floors/ceilings)
      - 40% Local LLM Semantic Evaluator (qualitative risk vectors & architectural failure modes)
    """
    rule_res = _evaluate_statutory_rule_risk(user_text, domain)
    rule_score = rule_res["rule_score"]
    statutory_floor = rule_res["statutory_floor"]
    statutory_ceiling = rule_res["statutory_ceiling"]

    model_res = {"success": False, "identified_risk_vectors": []}
    if use_model:
        model_res = _evaluate_model_semantic_risk(user_text, domain)

    if model_res.get("success") and model_res.get("model_risk_score") is not None:
        model_score = model_res["model_risk_score"]
        # Hybrid Ensemble Blend: 60% Rule + 40% Model
        blended_score = 0.60 * rule_score + 0.40 * model_score
        methodology = f"Hybrid Ensemble (60% Statutory Rule: {rule_score}% | 40% LLM Model: {model_score}%)"
    else:
        blended_score = float(rule_score)
        methodology = f"Statutory Rule Governance ({rule_score}%)"

    # Enforce statutory stage-gate floors and ceilings
    if statutory_floor is not None:
        blended_score = max(float(statutory_floor), blended_score)
    if statutory_ceiling is not None:
        blended_score = min(float(statutory_ceiling), blended_score)

    final_score = int(round(max(18, min(98, blended_score))))

    # Multi-tier Granular Risk Bands
    if final_score >= 85:
        risk_level = "CRITICAL"
        color = "#EF4444"
        summary = "Mission-critical financial tier. Direct automated capital risk or physical cash exposure requiring zero-trust stage gates."
    elif final_score >= 68:
        risk_level = "HIGH"
        color = "#F97316"
        summary = "High-impact regulated tier. Live consumer payment or digital credit facility governed by continuous statutory security controls."
    elif final_score >= 50:
        risk_level = "MEDIUM"
        color = "#F59E0B"
        summary = "Moderate operational tier. Asynchronous reconciliation, compliance reporting, or account aggregator data streaming."
    elif final_score >= 35:
        risk_level = "MODERATE / LOW"
        color = "#10A37F"
        summary = "Controlled trial tier. Regulatory sandbox, developer simulation, or bounded pilot cohort with synthetic capital."
    else:
        risk_level = "MINIMAL"
        color = "#059669"
        summary = "Non-transactional tier. Read-only audit viewer, internal analytics dashboard, or developer simulation."

    # Combine factors: statutory factors + model-derived risk vectors
    combined_factors = list(rule_res["factors"])
    if model_res.get("identified_risk_vectors"):
        for vec in model_res["identified_risk_vectors"]:
            combined_factors.append(f"Model Architectural Risk Vector: {vec}")

    return {
        "risk_level": risk_level,
        "risk_score_pct": final_score,
        "risk_color": color,
        "risk_summary": summary,
        "risk_factors": combined_factors,
        "factor_breakdown": rule_res["factor_breakdown"],
        "methodology": methodology,
        "statutory_rules": {
            "rule_score": rule_score,
            "statutory_floor": statutory_floor,
            "statutory_ceiling": statutory_ceiling
        },
        "model_evaluation": model_res
    }
