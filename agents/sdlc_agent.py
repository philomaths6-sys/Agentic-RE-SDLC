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

from agents.model_manager import get_sdlc_model


def recommend_sdlc_structured(requirement: str, rag_context: str = "") -> Dict[str, Any]:
    """Generate a structured SDLC recommendation dict for the given requirement.

    Returns a dict with keys:
        recommended_model, reasoning, alternatives_considered
    """
    if not input_is_safe(requirement):
        raise ValueError("Input rejected by guardrails.")

    tokenizer, model = get_sdlc_model()

    system_prompt = (
        "You are the SDLC Recommendation Agent for Indian financial institutions regulated by RBI, NPCI, UIDAI, SEBI, and DPDP Act 2023. "
        "Analyze the given requirement and recommend the optimal SDLC model. "
        "Explain WHY this model fits and list alternatives you considered but rejected with clear reasons. "
        "Output ONLY valid JSON."
    )

    # Retrieve dedicated SDLC regulatory governance context from Qdrant
    sdlc_context = ""
    try:
        from rag.retriever import retrieve
        hits = retrieve(f"regulatory SDLC lifecycle model applicability {requirement[:100]}", top_k=3)
        sdlc_context = "\n\n".join([
            f"[{h.get('citation', 'SDLC Governance')}]:\n{h.get('text', '').strip()[:450]}"
            for h in hits if h.get("text")
        ])
    except Exception as exc:
        pass

    combined_rag = f"{rag_context}\n\n{sdlc_context}".strip()
    rag_section = f"\n\nApplicable Regulatory & SDLC Governance Directives:\n{combined_rag[:1200]}" if combined_rag else ""

    user_prompt = f"""Analyze this requirement and recommend the best SDLC model. Explain your reasoning and why you rejected alternatives:

Requirement:
\"{requirement[:2000]}\"
{rag_section}

Output ONLY valid JSON in this exact structure:
{{
  "recommended_model": "Model Name (e.g., Agile-DevSecOps Hybrid / V-Model / V-Model with DevSecOps Gates / Waterfall / Prototyping Model / Iterative Model / Spiral Model / RAD / Incremental Model / Dual-Track Agile)",
  "reasoning": "Detailed explanation of why this SDLC model is the best fit for this specific requirement.",
  "alternatives_considered": [
    {{"model": "Alternative Model 1", "why_not": "Why this alternative was rejected."}},
    {{"model": "Alternative Model 2", "why_not": "Why this alternative was rejected."}}
  ]
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
            max_new_tokens=500,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    allowed, cleaned, reason = filter_output(response_text)
    if not allowed:
        raise ValueError(f"Output rejected by guardrails: {reason}")

    # Parse JSON robustly
    result = _parse_sdlc_json(cleaned)
    return result


def _parse_sdlc_json(text: str) -> Dict[str, Any]:
    """Robustly extract and parse SDLC JSON from model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try to fix truncated JSON
    if start != -1:
        for suffix in ["]}}", "}]}", "}}", "}"]:
            try:
                return json.loads(text[start:] + suffix)
            except Exception:
                continue

    # Regex fallback for malformed output
    def _re_extract(pattern, default=""):
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else default

    sdlc_name = _re_extract(r'"recommended_model"\s*:\s*"([^"]+)"', "Agile-DevSecOps Hybrid")
    reasoning = _re_extract(r'"reasoning"\s*:\s*"([^"]+)"', "")

    return {
        "recommended_model": sdlc_name,
        "reasoning": reasoning or f"Selected {sdlc_name} based on Indian regulatory requirements.",
        "alternatives_considered": [
            {"model": "Common Alternative", "why_not": "Alternative models lack the regulatory compliance structure required for this requirement."}
        ]
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
