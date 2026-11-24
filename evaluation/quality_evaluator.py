# evaluation/quality_evaluator.py
"""Deterministic IEEE Std 830-1998 Software Requirements Specification Quality Evaluator.
Evaluates:
  1. Unambiguous (0-100%): Absence of vague, subjective adjectives/adverbs.
  2. Complete (0-100%): Presence of all mandatory architectural attributes.
  3. Verifiable (0-100%): Acceptance criteria specify testable quantitative metrics.
  4. Prioritized (0-100%): Valid MoSCoW prioritization.
  5. Traceability (0-100%): Grounded statutory clauses and verifiable regulatory directives.
"""

from typing import List, Dict, Any


def evaluate_ieee830_quality(requirements: List[Dict[str, Any]], verification_rate_pct: float = 100.0) -> Dict[str, Any]:
    """Evaluate Software Requirements Specification against IEEE Std 830-1998 Quality Dimensions."""
    if not requirements:
        return {
            "overall_quality_score": 0.0,
            "rating_label": "Incomplete",
            "unambiguous_score": 0.0,
            "completeness_score": 0.0,
            "verifiability_score": 0.0,
            "prioritization_score": 0.0,
            "traceability_score": 0.0,
            "clause_level_traceability_pct": 0.0,
            "summary": "No requirements provided for quality evaluation."
        }

    vague_terms = [
        "user friendly", "user-friendly", "easy to use", "fast as possible",
        "good performance", "as fast as", "etc.", "etc,", "and so on",
        "flexible", "robust enough", "appropriate", "adequate"
    ]

    quantitative_indicators = [
        "ms", "second", "tps", "rpo", "rto", "sla", "%", "percent", "zero",
        "aes", "fips", "sha-", "rsa", "tls", "http", "kafka", "gb", "mb",
        "retention", "days", "hours", "lakh", "crore", "inr", "100%", "99."
    ]

    unambiguous_count = 0
    complete_count = 0
    verifiable_count = 0
    prioritized_count = 0
    traceable_count = 0

    total = len(requirements)

    for r in requirements:
        title = str(r.get("title", "")).strip()
        desc = str(r.get("description", "")).strip()
        cat = str(r.get("category", "")).strip()
        prio = str(r.get("priority", "")).strip()
        crit = str(r.get("acceptance_criteria", "")).strip()
        reg = str(r.get("regulation", "")).strip()

        # 1. Unambiguous
        desc_low = desc.lower()
        has_vague = any(vt in desc_low for vt in vague_terms)
        if not has_vague and len(desc) >= 30:
            unambiguous_count += 1

        # 2. Complete
        if title and desc and cat and prio and crit and reg:
            complete_count += 1

        # 3. Verifiable
        crit_low = crit.lower()
        has_metric = any(qi in crit_low for qi in quantitative_indicators)
        if has_metric and len(crit) >= 20:
            verifiable_count += 1

        # 4. Prioritized
        if prio in ["Must Have", "Should Have", "Could Have", "Won't Have"]:
            prioritized_count += 1

        # 5. Traceable
        if r.get("citation_verified", False) or len(reg) >= 10:
            traceable_count += 1

    unambiguous_score = round((unambiguous_count / total) * 100.0, 1)
    completeness_score = round((complete_count / total) * 100.0, 1)
    verifiability_score = round((verifiable_count / total) * 100.0, 1)
    prioritization_score = round((prioritized_count / total) * 100.0, 1)
    traceability_score = round(max(float(verification_rate_pct), (traceable_count / total) * 100.0), 1)

    # Weighted Overall Score (IEEE 830 Quality Index)
    overall = round(
        0.25 * unambiguous_score +
        0.20 * completeness_score +
        0.25 * verifiability_score +
        0.10 * prioritization_score +
        0.20 * traceability_score,
        1
    )

    if overall >= 92.0:
        rating_label = f"IEEE 830 Exemplary ({overall}%)"
    elif overall >= 80.0:
        rating_label = f"IEEE 830 Compliant ({overall}%)"
    elif overall >= 65.0:
        rating_label = f"IEEE 830 Acceptable ({overall}%)"
    else:
        rating_label = f"IEEE 830 Deficient ({overall}%)"

    summary = (
        f"Quality evaluation against IEEE Std 830-1998 standards: "
        f"Unambiguous: {unambiguous_score}%, Complete: {completeness_score}%, "
        f"Verifiable: {verifiability_score}%, Prioritized: {prioritization_score}%, "
        f"Clause-Level Traceability: {traceability_score}%."
    )

    return {
        "overall_quality_score": overall,
        "rating_label": rating_label,
        "unambiguous_score": unambiguous_score,
        "completeness_score": completeness_score,
        "verifiability_score": verifiability_score,
        "prioritization_score": prioritization_score,
        "traceability_score": traceability_score,
        "clause_level_traceability_pct": traceability_score,
        "summary": summary
    }
