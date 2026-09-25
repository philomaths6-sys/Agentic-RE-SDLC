# rag/evaluate_rag.py
"""Automated RAG Pipeline Evaluation Suite.

Evaluates:
  1. Retrieval Relevance & Top-K Hit Quality across Indian financial domains.
  2. Latency & Throughput (ms per query).
  3. Signal-to-Noise Ratio (absence of blank templates, presence of statutory clauses).
"""

import time
from typing import List, Dict, Any
from rag.retriever import retrieve

TEST_CASES = [
    {
        "domain": "Algorithmic Trading & Colocation",
        "query": "SEBI CSCRF algorithmic trading colocation cyber resilience latency",
        "expected_keywords": ["sebi", "cscrf", "cyber", "trading", "broker"],
        "expected_citation_match": "sebi"
    },
    {
        "domain": "ATM Physical & Cyber Security",
        "query": "ATM physical vault vibration tilt sensor tamper EPP zeroization PCI PTS",
        "expected_keywords": ["atm", "sensor", "vibration", "tamper", "pci", "rbi"],
        "expected_citation_match": "atm"
    },
    {
        "domain": "Digital Lending & KYC",
        "query": "RBI digital lending guidelines borrower Key Fact Statement KFS APR",
        "expected_keywords": ["rbi", "lending", "loan", "borrower", "kfs", "kyc"],
        "expected_citation_match": "rbi"
    },
    {
        "domain": "Data Privacy & Data Localization",
        "query": "Digital Personal Data Protection Act DPDP sovereign data localization on-soil",
        "expected_keywords": ["dpdp", "data", "personal", "processing", "protection"],
        "expected_citation_match": "dpdp"
    },
    {
        "domain": "SDLC Regulatory Governance",
        "query": "SDLC model compliance RBI SEBI V-model devsecops phase gates",
        "expected_keywords": ["sdlc", "model", "v-model", "devsecops", "regulatory", "gates"],
        "expected_citation_match": "sdlc"
    },
    {
        "domain": "Retail Payments & UPI",
        "query": "NPCI UPI AutoPay recurring mandate digital payment security controls",
        "expected_keywords": ["upi", "npci", "payment", "mandate", "security"],
        "expected_citation_match": "upi"
    }
]


def run_rag_evaluation(top_k: int = 3) -> Dict[str, Any]:
    print("=" * 70)
    print("🚀 STARTING AUTOMATED RAG PIPELINE EVALUATION")
    print("=" * 70)

    results = []
    total_latency_ms = 0.0
    passed_tests = 0

    for idx, tc in enumerate(TEST_CASES, 1):
        domain = tc["domain"]
        query = tc["query"]
        expected_kw = tc["expected_keywords"]
        expected_citation = tc["expected_citation_match"]

        start_time = time.perf_counter()
        hits = retrieve(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        total_latency_ms += elapsed_ms

        # 1. Relevance Assessment
        matched_chunks = 0
        found_keywords = set()
        noise_detected = False
        top_citation = hits[0].get("citation", "Unknown") if hits else "None"
        top_score = hits[0].get("score", 0.0) if hits else 0.0

        for h in hits:
            text = (h.get("text") or "").lower()
            # Check for noise
            if "<project name>" in text or "<author>" in text:
                noise_detected = True
            # Keyword presence
            for kw in expected_kw:
                if kw in text:
                    found_keywords.add(kw)
            matched_chunks += 1

        keyword_coverage = (len(found_keywords) / len(expected_kw)) * 100 if expected_kw else 100
        citation_relevant = expected_citation.lower() in top_citation.lower()
        test_passed = (keyword_coverage >= 50 or citation_relevant) and not noise_detected

        if test_passed:
            passed_tests += 1

        status_str = "✅ PASS" if test_passed else "❌ FAIL"

        results.append({
            "test_id": idx,
            "domain": domain,
            "top_citation": top_citation,
            "top_score": round(top_score, 4),
            "latency_ms": round(elapsed_ms, 2),
            "keyword_coverage_pct": round(keyword_coverage, 1),
            "noise_free": not noise_detected,
            "status": status_str
        })

        print(f"[{idx}/{len(TEST_CASES)}] {status_str} | {domain}")
        print(f"      Top Hit: {top_citation} (Score: {top_score:.4f}, Latency: {elapsed_ms:.1f}ms)")
        print(f"      Keyword Coverage: {keyword_coverage:.1f}% | Noise Free: {not noise_detected}")
        print("-" * 70)

    avg_latency = total_latency_ms / len(TEST_CASES)
    overall_accuracy = (passed_tests / len(TEST_CASES)) * 100

    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print(f"Total Test Cases:       {len(TEST_CASES)}")
    print(f"Passed:                 {passed_tests}/{len(TEST_CASES)} ({overall_accuracy:.1f}%)")
    print(f"Average Query Latency:  {avg_latency:.2f} ms")
    print("=" * 70)

    return {
        "overall_accuracy_pct": overall_accuracy,
        "avg_latency_ms": avg_latency,
        "passed_tests": passed_tests,
        "total_tests": len(TEST_CASES),
        "details": results
    }


if __name__ == "__main__":
    run_rag_evaluation()
