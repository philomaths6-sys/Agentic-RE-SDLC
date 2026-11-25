# agents/revision_node.py
"""True LLM Requirements Revision & Targeted Regeneration Node.

Uses the local GPU instruction-tuned model (Qwen-2.5-3B) to autonomously rewrite and repair
flawed specifications based on actionable feedback from the Verifier & Critic Agent.
"""

import re
import json
import torch
from typing import List, Dict, Any, Tuple
from agents.model_manager import get_base_model


def extract_json_resilient(text: str) -> Dict[str, Any]:
    """Robustly parse JSON even if truncated at max token limit."""
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except Exception:
            pass

    cleaned = text.strip()
    if "{" in cleaned:
        start_idx = cleaned.find("{")
        sub = cleaned[start_idx:]
        if sub.count('"') % 2 != 0:
            sub += '"'
        open_braces = sub.count("{") - sub.count("}")
        open_brackets = sub.count("[") - sub.count("]")
        sub += "]" * max(0, open_brackets)
        sub += "}" * max(0, open_braces)
        try:
            return json.loads(sub)
        except Exception:
            pass
    return {}


def revise_flawed_requirements(
    requirements: List[Dict[str, Any]],
    flawed_req_ids: List[str],
    regeneration_guidance: Dict[str, str],
    domain: str = "Indian Regulated Financial Infrastructure"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute autonomous LLM-powered architectural repair on flagged specifications."""
    flawed_set = set(flawed_req_ids)
    if not flawed_set:
        return requirements, []

    to_repair = [r for r in requirements if r.get("id") in flawed_set]
    if not to_repair:
        return requirements, []

    repair_blocks = []
    for r in to_repair:
        r_id = r.get("id")
        g = regeneration_guidance.get(r_id, "Harden quantitative acceptance criteria and remove ambiguity.")
        repair_blocks.append(
            f"[{r_id}]\n"
            f"Title: {r.get('title')}\n"
            f"Description: {r.get('description')}\n"
            f"Criteria: {r.get('acceptance_criteria')}\n"
            f"Critic Feedback: {g}"
        )

    system_prompt = (
        "You are an autonomous Senior Software Architect repairing financial system requirements. "
        "Resolve all critic feedback by enforcing concrete quantitative parameters (SLA %, latency in ms, "
        "TPS, AES-256-GCM, TLS 1.3) and eliminating vague or contradictory phrasing. "
        "Output strictly valid JSON with no markdown and no conversational preamble."
    )

    schema_example = chr(10).join([
        f'  "{r.get("id")}": {{"title": "<refined title>", "description": "<precise description>", "acceptance_criteria": "<testable quantitative criteria>"}},'
        for r in to_repair
    ]).rstrip(',')

    user_prompt = f"""Domain: {domain}

Flawed Specifications to Repair:
{chr(10).join(repair_blocks)}

Rewrite each specification to resolve the critic feedback. Output strictly JSON with this schema:
{{
{schema_example}
}}"""

    llm_repairs = {}
    try:
        tokenizer, model = get_base_model()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(600, len(to_repair) * 160 + 100),
                temperature=0.1,
                do_sample=False
            )
        raw_output = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        llm_repairs = extract_json_resilient(raw_output)
    except Exception as exc:
        print(f"[WARN] Revision Node LLM notice: {exc}")

    revised_requirements = []
    revision_records = []

    for r in requirements:
        r_id = r.get("id")
        if r_id not in flawed_set:
            revised_requirements.append(r)
            continue

        orig_title = str(r.get("title", ""))
        orig_desc = str(r.get("description", ""))
        orig_crit = str(r.get("acceptance_criteria", ""))
        guidance = regeneration_guidance.get(r_id, "")

        repair_data = llm_repairs.get(r_id, {})
        new_title = str(repair_data.get("title") or orig_title)
        new_desc = str(repair_data.get("description") or orig_desc)
        new_crit = repair_data.get("acceptance_criteria")

        if isinstance(new_crit, list):
            new_crit = "; ".join([f"{item.get('metric', '')}: {item.get('value', '')}" if isinstance(item, dict) else str(item) for item in new_crit])
        elif isinstance(new_crit, dict):
            new_crit = "; ".join([f"{k}: {v}" for k, v in new_crit.items()])
        elif not new_crit:
            new_crit = orig_crit
        else:
            new_crit = str(new_crit)

        revised_item = {
            **r,
            "title": new_title,
            "description": new_desc,
            "acceptance_criteria": new_crit,
            "revised": True,
            "revision_notes": guidance,
            "revision_mode": "True LLM Regeneration (Qwen-2.5-3B)"
        }
        revised_requirements.append(revised_item)

        revision_records.append({
            "req_id": r_id,
            "title": new_title,
            "guidance_applied": guidance,
            "before_desc": orig_desc[:120],
            "after_desc": new_desc[:120],
            "before_crit": orig_crit[:120],
            "after_crit": new_crit[:120],
            "model": "unsloth/Qwen2.5-3B-Instruct"
        })

    return revised_requirements, revision_records
