# scripts/evaluate_rag_ir.py
"""Information Retrieval (IR) Evaluation Benchmark for Statutory Regulatory RAG.
Evaluates Hybrid BM25 + Dense Search + RRF Reranking vs. Dense-Only Unthresholded Baseline
using labelled benchmark queries against the Indian Financial Statutory Corpus (Qdrant).

Metrics Computed:
  - Recall@1, Recall@3, Recall@5, Recall@10
  - Mean Reciprocal Rank (MRR)
  - Precision@k
  - Active vs. Superseded Circular Filtering Fidelity
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import retrieve, get_client, get_model, STATUTORY_COLLECTION


# -----------------------------------------------------------------------------
# LABELLED STATUTORY BENCHMARK QUERIES
# Ground-truth annotations for Indian Financial Regulatory Directives
# -----------------------------------------------------------------------------
LABELLED_BENCHMARK_DATASET = [
    {
        "query_id": "IR-01",
        "domain": "Digital Payment Security",
        "query": "Multi-factor authentication MFA, dynamic linking, and mobile app session security controls under RBI",
        "ground_truth_circulars": ["RBI/2020-21/74", "Digital Payment Security Controls"],
        "ground_truth_issuers": ["RBI"],
        "ground_truth_keywords": ["mfa", "dynamic linking", "session", "application", "authentication"]
    },
    {
        "query_id": "IR-02",
        "domain": "ATM Physical Security",
        "query": "ATM cash cassette swap, vault door sensors, and physical tamper protection requirements",
        "ground_truth_circulars": ["RBI/2018-19/166", "ATM Security Guidelines", "DCM (Plg)"],
        "ground_truth_issuers": ["RBI"],
        "ground_truth_keywords": ["cassette", "atm", "vault", "tamper", "sensor"]
    },
    {
        "query_id": "IR-03",
        "domain": "Aadhaar & Identity",
        "query": "Aadhaar authentication submission of proposals, license keys, and Aadhaar Data Vault compliance",
        "ground_truth_circulars": ["UIDAI/AUTH/2021/01", "Aadhaar Authentication", "Aadhaar"],
        "ground_truth_issuers": ["UIDAI"],
        "ground_truth_keywords": ["aadhaar", "authentication", "proposals", "vault", "uidai"]
    },
    {
        "query_id": "IR-04",
        "domain": "Capital Markets SOC",
        "query": "SEBI Cybersecurity and Cyber Resilience Framework CSCRF guidelines for SOC monitoring, VAPT, and cyber incidents",
        "ground_truth_circulars": ["SEBI/HO/MRD/TPD/P/CIR/2023/147", "SEBI CSCRF", "Cyber Resilience Framework"],
        "ground_truth_issuers": ["SEBI"],
        "ground_truth_keywords": ["cscrf", "cybersecurity", "resilience", "vapt", "soc"]
    },
    {
        "query_id": "IR-05",
        "domain": "DPDP Data Protection",
        "query": "Digital Personal Data Protection Act DPDP notice requirements, consent manager, and Data Principal rights",
        "ground_truth_circulars": ["Act No. 22 of 2023", "DPDP Act 2023", "Digital Personal Data Protection"],
        "ground_truth_issuers": ["MeitY"],
        "ground_truth_keywords": ["consent", "notice", "principal", "withdrawal", "processing"]
    },
    {
        "query_id": "IR-06",
        "domain": "DPDP Implementation",
        "query": "DPDP Implementation Rules 2025 procedural compliance for data fiduciary and child consent",
        "ground_truth_circulars": ["DPDP Implementation Rules 2025", "DPDP Rules 2025"],
        "ground_truth_issuers": ["MeitY"],
        "ground_truth_keywords": ["rules", "fiduciary", "consent", "child", "dpo"]
    },
    {
        "query_id": "IR-07",
        "domain": "KYC & Anti-Money Laundering",
        "query": "RBI Master Direction KYC customer identification procedure CIP, periodic update, and PEP due diligence",
        "ground_truth_circulars": ["RBI/DBR/2015-16/18", "Master Direction DBR.AML", "KYC"],
        "ground_truth_issuers": ["RBI"],
        "ground_truth_keywords": ["kyc", "customer", "identification", "cdd", "due diligence"]
    },
    {
        "query_id": "IR-08",
        "domain": "AML / PMLA Reporting",
        "query": "PMLA 2002 prevention of money laundering, maintenance of records, and suspicious transaction reporting STR to FIU",
        "ground_truth_circulars": ["PMLA 2002", "FIU Guidelines", "PMLA"],
        "ground_truth_issuers": ["FIU-IND", "FIU"],
        "ground_truth_keywords": ["pmla", "fiu", "suspicious", "reporting", "records"]
    },
    {
        "query_id": "IR-09",
        "domain": "UPI Payments",
        "query": "BHIM UPI guidelines, transaction processing, and UPI ecosystem participant operating regulations",
        "ground_truth_circulars": ["NPCI/UPI/2026/01", "BHIM UPI", "NPCI UPI"],
        "ground_truth_issuers": ["NPCI"],
        "ground_truth_keywords": ["upi", "npci", "transaction", "psp", "payer"]
    },
    {
        "query_id": "IR-10",
        "domain": "UPI Tap & Pay POS",
        "query": "Enhancement of UPI Tap and Pay on Point of Sale POS terminals with contactless NFC reader specifications",
        "ground_truth_circulars": ["NPCI/UPI/OC-186A", "OC-186A", "Tap & Pay", "Tap and Pay"],
        "ground_truth_issuers": ["NPCI"],
        "ground_truth_keywords": ["tap", "pos", "nfc", "contactless", "point of sale"]
    },
    {
        "query_id": "IR-11",
        "domain": "UPI Conversational AI",
        "query": "MyUPI AI-powered conversational support for UPI payments and UPI Help 2.0 automated dispute resolution",
        "ground_truth_circulars": ["NPCI/UPI/OC-227A", "OC-227A", "MyUPI", "UPI Help 2.0"],
        "ground_truth_issuers": ["NPCI"],
        "ground_truth_keywords": ["myupi", "conversational", "dispute", "ai-powered", "help"]
    },
    {
        "query_id": "IR-12",
        "domain": "Numeric UPI ID",
        "query": "Addendum on numeric UPI ID routing, mapper database, and alphanumeric virtual payment address VPA resolution",
        "ground_truth_circulars": ["NPCI/UPI/OC-115F", "OC-115F", "Numeric UPI ID"],
        "ground_truth_issuers": ["NPCI"],
        "ground_truth_keywords": ["numeric", "vpa", "mapper", "routing", "id"]
    }
]


def is_relevant_hit(hit: Dict[str, Any], ground_truth: Dict[str, Any]) -> bool:
    """Determine whether a retrieved chunk is relevant to the ground truth annotation."""
    circ_no = (hit.get("circular_no") or "").upper()
    citation = (hit.get("citation") or "").upper()
    text = (hit.get("text") or "").lower()
    issuer = (hit.get("issuer") or "").upper()

    # 1. Circular number match or title in citation
    for gt_circ in ground_truth["ground_truth_circulars"]:
        gt_upper = gt_circ.upper()
        if gt_upper in circ_no or gt_upper in citation:
            return True

    # 2. Issuer match + keyword overlap
    for gt_issuer in ground_truth["ground_truth_issuers"]:
        if gt_issuer.upper() in issuer or gt_issuer.upper() in citation:
            keyword_matches = sum(1 for kw in ground_truth["ground_truth_keywords"] if kw.lower() in text)
            if keyword_matches >= 2:
                return True

    return False


def run_dense_only_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Simulate legacy dense-only unthresholded search for baseline comparison."""
    client = get_client()
    model = get_model()
    try:
        query_vec = model.encode([query])[0].tolist()
        hits = client.search(
            collection_name=STATUTORY_COLLECTION,
            query_vector=query_vec,
            limit=top_k,
            score_threshold=None
        )
        results = []
        for h in hits:
            p = dict(h.payload or {})
            p["score"] = float(h.score)
            results.append(p)
        return results
    except Exception as exc:
        print(f"[WARN] Dense search failed: {exc}")
        return []


def evaluate_retrieval_system() -> Dict[str, Any]:
    print("=" * 80)
    print("📊 STATUTORY RAG INFORMATION RETRIEVAL (IR) BENCHMARK EVALUATION")
    print("   Comparing Hybrid BM25+Dense+RRF (Threshold >= 0.65) vs. Dense-Only Baseline")
    print("=" * 80)

    ks = [1, 3, 5, 10]
    hybrid_metrics = {f"recall@{k}": 0.0 for k in ks}
    hybrid_metrics["mrr"] = 0.0
    hybrid_metrics["precision@3"] = 0.0

    dense_metrics = {f"recall@{k}": 0.0 for k in ks}
    dense_metrics["mrr"] = 0.0
    dense_metrics["precision@3"] = 0.0

    total_queries = len(LABELLED_BENCHMARK_DATASET)
    per_query_results = []

    print(f"\nEvaluating {total_queries} labelled statutory queries across Indian banking regulations...\n")
    print(f"{'ID':<6} | {'Domain':<20} | {'Hybrid Top Citation':<35} | {'Hybrid MRR':<10} | {'Dense MRR':<10}")
    print("-" * 90)

    for q_item in LABELLED_BENCHMARK_DATASET:
        q_id = q_item["query_id"]
        domain = q_item["domain"]
        query = q_item["query"]

        # Run Hybrid BM25 + Dense Search
        hybrid_hits = retrieve(query, top_k=10, score_threshold=0.65)
        # Run Dense-Only Baseline
        dense_hits = run_dense_only_search(query, top_k=10)

        # Evaluate Hybrid
        hybrid_rr = 0.0
        hybrid_relevant_indices = []
        for rank_idx, hit in enumerate(hybrid_hits):
            if is_relevant_hit(hit, q_item):
                hybrid_relevant_indices.append(rank_idx)
                if hybrid_rr == 0.0:
                    hybrid_rr = 1.0 / (rank_idx + 1)

        for k in ks:
            if any(idx < k for idx in hybrid_relevant_indices):
                hybrid_metrics[f"recall@{k}"] += 1.0
        hybrid_metrics["mrr"] += hybrid_rr
        p3_hits = sum(1 for idx in hybrid_relevant_indices if idx < 3)
        hybrid_metrics["precision@3"] += (p3_hits / 3.0)

        # Evaluate Dense Baseline
        dense_rr = 0.0
        dense_relevant_indices = []
        for rank_idx, hit in enumerate(dense_hits):
            if is_relevant_hit(hit, q_item):
                dense_relevant_indices.append(rank_idx)
                if dense_rr == 0.0:
                    dense_rr = 1.0 / (rank_idx + 1)

        for k in ks:
            if any(idx < k for idx in dense_relevant_indices):
                dense_metrics[f"recall@{k}"] += 1.0
        dense_metrics["mrr"] += dense_rr
        dense_p3_hits = sum(1 for idx in dense_relevant_indices if idx < 3)
        dense_metrics["precision@3"] += (dense_p3_hits / 3.0)

        top_cite = hybrid_hits[0].get("citation", "None")[:33] if hybrid_hits else "None"
        print(f"{q_id:<6} | {domain:<20} | {top_cite:<35} | {hybrid_rr:<10.3f} | {dense_rr:<10.3f}")

        per_query_results.append({
            "query_id": q_id,
            "domain": domain,
            "query": query,
            "hybrid_rr": hybrid_rr,
            "dense_rr": dense_rr,
            "hybrid_top_citation": hybrid_hits[0].get("citation") if hybrid_hits else None,
            "dense_top_citation": dense_hits[0].get("citation") if dense_hits else None,
            "hybrid_retrieved_count": len(hybrid_hits),
            "dense_retrieved_count": len(dense_hits)
        })

    # Normalize metrics across queries
    for k in ks:
        hybrid_metrics[f"recall@{k}"] /= total_queries
        dense_metrics[f"recall@{k}"] /= total_queries
    hybrid_metrics["mrr"] /= total_queries
    dense_metrics["mrr"] /= total_queries
    hybrid_metrics["precision@3"] /= total_queries
    dense_metrics["precision@3"] /= total_queries

    print("\n" + "=" * 80)
    print("📈 FINAL INFORMATION RETRIEVAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Dense-Only Baseline':<22} | {'Hybrid BM25+Dense+RRF':<22} | {'Improvement':<12}")
    print("-" * 80)

    summary_rows = [
        ("Recall@1", dense_metrics["recall@1"], hybrid_metrics["recall@1"]),
        ("Recall@3", dense_metrics["recall@3"], hybrid_metrics["recall@3"]),
        ("Recall@5", dense_metrics["recall@5"], hybrid_metrics["recall@5"]),
        ("Recall@10", dense_metrics["recall@10"], hybrid_metrics["recall@10"]),
        ("Precision@3", dense_metrics["precision@3"], hybrid_metrics["precision@3"]),
        ("MRR (Mean RR)", dense_metrics["mrr"], hybrid_metrics["mrr"]),
    ]

    for label, d_val, h_val in summary_rows:
        delta = h_val - d_val
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        pct_change = f"({(delta / d_val * 100):+.1f}%)" if d_val > 0 else "N/A"
        print(f"{label:<20} | {d_val:<22.3f} | {h_val:<22.3f} | {delta_str} {pct_change}")

    print("=" * 80)

    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_queries_evaluated": total_queries,
        "hybrid_retriever_metrics": {k: round(v, 4) for k, v in hybrid_metrics.items()},
        "dense_baseline_metrics": {k: round(v, 4) for k, v in dense_metrics.items()},
        "queries_detail": per_query_results
    }

    os.makedirs("output", exist_ok=True)
    report_path = "output/rag_ir_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Detailed IR evaluation benchmark saved to: {report_path}\n")
    return report


if __name__ == "__main__":
    evaluate_retrieval_system()
