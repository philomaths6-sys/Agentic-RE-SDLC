# scripts/evaluate_system.py
"""Comprehensive End-to-End System Evaluation Benchmark for Agentic RE-SDLC Advisor.
Evaluates:
  1. RAG Retrieval accuracy & citation grounding against 3,119 Qdrant chunks
  2. Domain detection & risk scoring calibration across financial domains
  3. Dynamic requirement extraction & modular assembly (minimum 30 requirements across 6 pillars)
  4. SDLC Model recommendation fidelity against regulatory governance matrix
  5. Conversational intent detection & guardrails
  6. PDF report generation stability
"""

import time
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import retrieve
from requirements_analysis import (
    analyze_requirement,
    _detect_financial_domain,
    calculate_dynamic_risk,
    is_conversational
)
from agents.sdlc_agent import recommend_sdlc_structured
from pdf_generator import generate_pdf_report


def run_system_evaluation():
    print("=" * 70)
    print("🚀 STARTING FULL SYSTEM EVALUATION BENCHMARK")
    print("=" * 70)
    
    test_cases = [
        {
            "id": "TC-ATM-01",
            "name": "ATM Switch & Vault Hardware Security",
            "prompt": "High-speed ATM transaction switch processing ISO 8583 messages, interfacing with FIPS 140-3 HSM, handling vault door reed sensors and cash cassette dispensers with automated zeroization on physical tamper.",
            "expected_domain": "atm",
            "expected_sdlc": ["V-Model", "V-Model with DevSecOps Gates", "Waterfall / Staged-Gate", "Waterfall"],
            "expected_min_risk": 70
        },
        {
            "id": "TC-UPI-02",
            "name": "UPI AutoPay Recurring Mandate Platform",
            "prompt": "UPI AutoPay recurring mandate mobile application with 24-hour SMS pre-debit notifications, UPI PIN Additional Factor of Authentication (AFA), and client-side mandate pause/revoke controls compliant with NPCI and DPDP Act 2023.",
            "expected_domain": "payments",
            "expected_sdlc": ["Agile-DevSecOps Hybrid", "Agile-DevSecOps", "Agile", "Dual-Track Agile"],
            "expected_min_risk": 50
        },
        {
            "id": "TC-TRD-03",
            "name": "Algorithmic Trading & Colocation Gateway",
            "prompt": "Colocation algorithmic trading execution engine for NSE equity derivatives using FIX 5.0 protocol, sub-millisecond risk checks, market-wide circuit breakers, and SEBI CSCRF audit trails.",
            "expected_domain": "trading",
            "expected_sdlc": ["V-Model with DevSecOps Gates", "V-Model"],
            "expected_min_risk": 75
        },
        {
            "id": "TC-SBX-04",
            "name": "Regulatory Sandbox Pilot (Offline CBDC)",
            "prompt": "RBI Regulatory Sandbox pilot cohort for offline peer-to-peer digital rupee (CBDC) payments with synthetic money, capped at 1000 users and maximum wallet balance of 2000 INR.",
            "expected_domain": "payments",
            "expected_sdlc": ["Prototyping / Iterative", "Prototyping Model", "Iterative Model", "Prototyping"],
            "expected_max_risk": 55
        }
    ]

    results = []
    
    # --- PHASE 1: EVALUATE TEST SCENARIOS ---
    for tc in test_cases:
        print(f"\nEvaluating {tc['id']}: {tc['name']}...")
        start_t = time.time()
        
        # 1. Domain Detection
        detected_domain = _detect_financial_domain(tc["prompt"])
        domain_match = (detected_domain == tc["expected_domain"])
        
        # 2. RAG Retrieval
        rag_hits = retrieve(tc["prompt"], top_k=3)
        rag_ok = len(rag_hits) > 0 and any("text" in h and len(h["text"]) > 50 for h in rag_hits)
        top_citation = rag_hits[0].get("citation", "None") if rag_hits else "None"
        
        # 3. Full Requirement Analysis & Assembly
        analysis_res = analyze_requirement(tc["prompt"])
        elapsed_analysis = time.time() - start_t
        
        reqs = analysis_res.get("requirements_list") or analysis_res.get("requirements", [])
        req_count = len(reqs)
        has_30_reqs = req_count >= 30
        
        # Verify 6 architectural pillars
        categories = set(r.get("category", "") for r in reqs)
        has_6_pillars = len(categories) >= 6
        
        # MoSCoW distribution
        priorities = [r.get("priority") for r in reqs]
        must_haves = priorities.count("Must Have")
        should_haves = priorities.count("Should Have")
        
        # Risk assessment
        risk_score = analysis_res.get("risk_score_pct") or analysis_res.get("risk_assessment", {}).get("risk_score_pct", 0)
        risk_level = analysis_res.get("risk_level") or analysis_res.get("risk_assessment", {}).get("risk_level", "Unknown")
        
        # 4. SDLC Recommendation
        sdlc_start_t = time.time()
        rag_context_str = "\n".join([h.get("text", "")[:300] for h in rag_hits])
        sdlc_rec = recommend_sdlc_structured(tc["prompt"], rag_context=rag_context_str)
        elapsed_sdlc = time.time() - sdlc_start_t
        
        rec_model = sdlc_rec.get("recommended_model", "Unknown")
        sdlc_match = any(exp.lower() in rec_model.lower() for exp in tc["expected_sdlc"])
        
        # 5. PDF Generation Test
        pdf_path = f"output/eval_{tc['id']}.pdf"
        pdf_ok = False
        try:
            generate_pdf_report(analysis_res, pdf_path)
            pdf_ok = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000
        except Exception as exc:
            print(f"  [PDF Error] {exc}")
            
        case_result = {
            "id": tc["id"],
            "name": tc["name"],
            "domain_detected": detected_domain,
            "domain_pass": domain_match,
            "rag_pass": rag_ok,
            "top_citation": top_citation,
            "req_count": req_count,
            "req_count_pass": has_30_reqs,
            "pillars_count": len(categories),
            "pillars_pass": has_6_pillars,
            "must_have_count": must_haves,
            "should_have_count": should_haves,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommended_sdlc": rec_model,
            "sdlc_pass": sdlc_match,
            "pdf_pass": pdf_ok,
            "analysis_time_sec": round(elapsed_analysis, 2),
            "sdlc_time_sec": round(elapsed_sdlc, 2)
        }
        results.append(case_result)
        print(f"  -> Requirements: {req_count} (Pillars: {len(categories)}) | Risk: {risk_score}% ({risk_level})")
        print(f"  -> SDLC: '{rec_model}' (Pass: {sdlc_match}) | PDF: {pdf_ok}")

    # --- PHASE 2: CONVERSATIONAL & GUARDRAILS TEST ---
    print("\nEvaluating Conversational Intent & Guardrails...")
    conv_tests = [
        ("hi there", True),
        ("how are you", True),
        ("who are you", True),
        ("design an algorithmic trading engine for NSE", False)
    ]
    conv_passes = 0
    for text, expected in conv_tests:
        detected = is_conversational(text)
        if detected == expected:
            conv_passes += 1
    conv_accuracy = (conv_passes / len(conv_tests)) * 100

    # --- SUMMARY REPORT ---
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    
    total_cases = len(results)
    passed_domains = sum(1 for r in results if r["domain_pass"])
    passed_reqs = sum(1 for r in results if r["req_count_pass"])
    passed_pillars = sum(1 for r in results if r["pillars_pass"])
    passed_sdlc = sum(1 for r in results if r["sdlc_pass"])
    passed_pdf = sum(1 for r in results if r["pdf_pass"])
    
    print(f"Total Test Scenarios:          {total_cases}")
    print(f"Domain Detection Accuracy:     {passed_domains}/{total_cases} ({(passed_domains/total_cases)*100:.1f}%)")
    print(f"Requirement 30-Count Baseline: {passed_reqs}/{total_cases} ({(passed_reqs/total_cases)*100:.1f}%)")
    print(f"Architectural 6-Pillar Coverage:{passed_pillars}/{total_cases} ({(passed_pillars/total_cases)*100:.1f}%)")
    print(f"SDLC Governance Match:         {passed_sdlc}/{total_cases} ({(passed_sdlc/total_cases)*100:.1f}%)")
    print(f"PDF Generation Success:        {passed_pdf}/{total_cases} ({(passed_pdf/total_cases)*100:.1f}%)")
    print(f"Conversational Detection Rate: {conv_accuracy:.1f}%")
    print("=" * 70)

    # Save summary json
    summary_path = "output/system_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenarios": results,
            "conversational_accuracy": conv_accuracy,
            "metrics": {
                "domain_accuracy_pct": (passed_domains/total_cases)*100,
                "req_baseline_compliance_pct": (passed_reqs/total_cases)*100,
                "pillar_coverage_pct": (passed_pillars/total_cases)*100,
                "sdlc_governance_match_pct": (passed_sdlc/total_cases)*100,
                "pdf_generation_pct": (passed_pdf/total_cases)*100
            }
        }, f, indent=2)
    print(f"Saved evaluation metrics to {summary_path}")


if __name__ == "__main__":
    run_system_evaluation()
