# rag/chunker.py
"""Section- and Clause-Aware Statutory Document Chunker.

Extracts meaningful statutory units (clauses, sections, rules, articles)
rather than naive fixed-character splits.
Annotates every chunk with rich metadata:
- issuer (RBI, SEBI, NPCI, UIDAI, MeitY/DPDP, FIU-IND)
- circular_no
- circular_date
- status (ACTIVE / SUPERSEDED)
- clause_ref
- section_title
- citation (authoritative legal string)
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Canonical metadata registry for known regulatory files
REGULATORY_CATALOG = {
    "rbi_master_direction_dated_18.02.2021_-_digital_payment_security_controls.pdf": {
        "issuer": "RBI",
        "circular_no": "RBI/2020-21/74 DOR.ATMs.REC.No.8/02.01.001/2020-21",
        "circular_date": "2021-02-18",
        "title": "Master Direction on Digital Payment Security Controls",
        "status": "ACTIVE",
        "version": "2021.1",
        "domain": "payments"
    },
    "558308370kycunderrbimasterdirection-quicksummary-7.7.2023.pdf": {
        "issuer": "RBI",
        "circular_no": "RBI/DBR/2015-16/18 Master Direction DBR.AML.BC.No.81/14.01.001/2015-16",
        "circular_date": "2023-07-07",
        "title": "Master Direction - Know Your Customer (KYC) Direction, 2016",
        "status": "ACTIVE",
        "version": "2023.2",
        "domain": "kyc"
    },
    "rbi_atm_security_guidelines.md": {
        "issuer": "RBI",
        "circular_no": "RBI/2018-19/166 DCM (Plg) No.2968/10.25.007/2018-19",
        "circular_date": "2019-04-11",
        "title": "Security Measures for ATMs & Currency Chests",
        "status": "ACTIVE",
        "version": "2019.1",
        "domain": "atm"
    },
    "sebi_cscrf_circular_2023.pdf": {
        "issuer": "SEBI",
        "circular_no": "SEBI/HO/MRD/TPD/P/CIR/2023/147",
        "circular_date": "2023-08-11",
        "title": "Cybersecurity & Cyber Resilience Framework (CSCRF) for Stock Brokers",
        "status": "ACTIVE",
        "version": "2023.1",
        "domain": "trading"
    },
    "dpdp_act_2023_official_gazette.pdf": {
        "issuer": "MeitY",
        "circular_no": "Act No. 22 of 2023",
        "circular_date": "2023-08-11",
        "title": "Digital Personal Data Protection Act, 2023",
        "status": "ACTIVE",
        "version": "2023.1",
        "domain": "privacy"
    },
    "dpdp_rules_2025_english_only.pdf": {
        "issuer": "MeitY",
        "circular_no": "DPDP Implementation Rules 2025",
        "circular_date": "2025-01-03",
        "title": "Digital Personal Data Protection Rules, 2025",
        "status": "ACTIVE",
        "version": "2025.1",
        "domain": "privacy"
    },
    "pmla_2002.pdf": {
        "issuer": "FIU-IND",
        "circular_no": "PMLA 2002 / FIU Guidelines",
        "circular_date": "2002-07-01",
        "title": "Prevention of Money Laundering Act & Reporting Directives",
        "status": "ACTIVE",
        "version": "2023-Amended",
        "domain": "compliance"
    },
    "bhim_upi_guidelines_2026_012a0b1bce.pdf": {
        "issuer": "NPCI",
        "circular_no": "NPCI/UPI/2026/01",
        "circular_date": "2026-01-15",
        "title": "BHIM UPI Operating Guidelines & Architecture Norms",
        "status": "ACTIVE",
        "version": "2026.1",
        "domain": "payments"
    },
    "upi-_-oc-no_-227a-_-fy-2026-27-_-addendum-to-upi-oc-227-–-“myupi”---ai-powered-support-for-upi-payments-(upi-help-2_0).pdf": {
        "issuer": "NPCI",
        "circular_no": "NPCI/UPI/OC-227A",
        "circular_date": "2026-04-10",
        "title": "UPI OC-227A: AI-powered support for UPI payments (UPI-HELP 2.0)",
        "status": "ACTIVE",
        "version": "2026.1",
        "domain": "payments"
    },
    "upi-_-oc-no_-186a-_-fy-2026-27-_-enhancement---upi-tap-&-pay-on-point-of-sale-(pos).pdf": {
        "issuer": "NPCI",
        "circular_no": "NPCI/UPI/OC-186A",
        "circular_date": "2026-03-12",
        "title": "UPI OC-186A: Enhancement - UPI Tap & Pay on PoS",
        "status": "ACTIVE",
        "version": "2026.1",
        "domain": "payments"
    },
    "upi-_-oc-no_-115f-_-fy-2026-27-_-addendum-to-oc-npci_upi_oc-115e_2024-25-on-the-numeric-upi-id.pdf": {
        "issuer": "NPCI",
        "circular_no": "NPCI/UPI/OC-115F",
        "circular_date": "2026-02-20",
        "title": "UPI OC-115F: Numeric UPI ID Addendum (Supersedes OC-115E)",
        "status": "ACTIVE",
        "supersedes": "NPCI/UPI/OC-115E",
        "version": "2026.1",
        "domain": "payments"
    },
    "application_form_guidelines_for_submission_of_proposals_for_use_of_aadhaar_authentication.pdf": {
        "issuer": "UIDAI",
        "circular_no": "UIDAI/AUTH/2021/01",
        "circular_date": "2021-09-01",
        "title": "UIDAI Aadhaar Authentication & Data Vault Guidelines",
        "status": "ACTIVE",
        "version": "2021.1",
        "domain": "kyc"
    },
    "fintech_sdlc_regulations.md": {
        "issuer": "RBI/SEBI/NPCI",
        "circular_no": "FinTech SDLC Governance Matrix",
        "circular_date": "2024-01-01",
        "title": "Statutory SDLC Governance & System Development Life Cycle Framework",
        "status": "ACTIVE",
        "version": "2024.1",
        "domain": "governance"
    }
}


# Heading & clause regexes
_CLAUSE_REGEX = re.compile(
    r"^(?:"
    r"(?:#{1,4}\s+(.+))|"
    r"((?:Section|Sec\.|Clause|Article|Rule|Chapter|Paragraph|Para)\s+[0-9A-Za-z\.\(\)\-]+(?:\s*[:\-—]\s*.+)?)|"
    r"([0-9]+\.[0-9]+(?:\.[0-9]+)?\s+[A-Z].*)|"
    r"(\([0-9a-z]+\)\s+[A-Z].*)"
    r")$",
    re.MULTILINE
)


def extract_document_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract or resolve metadata for a statutory or uploaded document."""
    fname_key = file_path.name.lower()
    meta = REGULATORY_CATALOG.get(fname_key, {})
    if meta:
        return {
            "source": str(file_path),
            "filename": file_path.name,
            "issuer": meta.get("issuer", "Statutory Authority"),
            "circular_no": meta.get("circular_no", file_path.stem),
            "circular_date": meta.get("circular_date", "2024"),
            "doc_title": meta.get("title", file_path.stem.replace("_", " ")),
            "status": meta.get("status", "ACTIVE"),
            "version": meta.get("version", "1.0"),
            "domain": meta.get("domain", "general"),
            "is_statutory": True
        }

    # Generic or uploaded user document
    is_user = "indian_docx" in str(file_path) or "uploads" in str(file_path)
    return {
        "source": str(file_path),
        "filename": file_path.name,
        "issuer": "Enterprise Document" if is_user else "Regulatory Archive",
        "circular_no": file_path.stem,
        "circular_date": "2026",
        "doc_title": file_path.stem.replace("_", " "),
        "status": "ACTIVE",
        "version": "1.0",
        "domain": "user_spec" if is_user else "general",
        "is_statutory": not is_user
    }


def chunk_by_clause_or_section(
    text: str,
    meta: Dict[str, Any],
    max_chunk_chars: int = 1400,
    min_chunk_chars: int = 150
) -> List[Dict[str, Any]]:
    """Segment document text into natural legal clauses/sections with structured metadata.
    
    Guarantees:
    - Never cuts clauses or paragraphs mid-sentence.
    - Preserves legal numbering (e.g. 'Section 4.1', 'Clause 8.2(b)').
    - Annotates every chunk with authoritative citation string.
    """
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    sections: List[Dict[str, Any]] = []
    curr_header = meta.get("doc_title", "General Provisions")
    curr_clause_no = "Clause 1.0"
    curr_buffer: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if curr_buffer:
                curr_buffer.append("\n")
            continue

        # Check if line matches a section or clause header
        match = _CLAUSE_REGEX.match(stripped)
        if match and len(curr_buffer) > 0:
            section_body = "".join(curr_buffer).strip()
            if len(section_body) >= min_chunk_chars:
                sections.append({
                    "header": curr_header,
                    "clause_no": curr_clause_no,
                    "text": section_body
                })
                curr_buffer = []

            # Update header and clause number
            header_text = match.group(1) or match.group(2) or match.group(3) or match.group(4) or stripped
            curr_header = header_text[:120].strip()
            # Extract clause numbering if present
            c_match = re.search(r"\b(?:Section|Sec\.|Clause|Article|Rule|Para)\s+([0-9A-Za-z\.\(\)\-]+)", curr_header, re.IGNORECASE)
            if c_match:
                curr_clause_no = c_match.group(0)
            elif re.match(r"^[0-9]+\.[0-9]+", curr_header):
                curr_clause_no = f"Clause {curr_header.split()[0]}"
            else:
                curr_clause_no = f"Section ({curr_header[:30]})"

        curr_buffer.append(line + "\n")

    if curr_buffer:
        section_body = "".join(curr_buffer).strip()
        if section_body:
            sections.append({
                "header": curr_header,
                "clause_no": curr_clause_no,
                "text": section_body
            })

    # If no structural clauses found, fallback to paragraph-level splitting
    if not sections:
        paras = re.split(r"\n\s*\n", text)
        for idx, p in enumerate(paras):
            pt = p.strip()
            if len(pt) >= 50:
                sections.append({
                    "header": meta.get("doc_title", "Document Section"),
                    "clause_no": f"Para {idx+1}",
                    "text": pt
                })

    # Refine oversized sections cleanly on sentence boundaries
    final_chunks: List[Dict[str, Any]] = []
    for s in sections:
        body = s["text"]
        if len(body) <= max_chunk_chars:
            citation_str = (
                f"{meta.get('issuer', 'Statutory')} {meta.get('doc_title', 'Circular')} "
                f"({meta.get('circular_date', '2024')}), {s['clause_no']} [{meta.get('status', 'ACTIVE')}]"
            )
            final_chunks.append({
                "text": f"[{s['clause_no']} - {s['header']}]\n{body}",
                "clause_no": s["clause_no"],
                "section_header": s["header"],
                "citation": citation_str,
                "issuer": meta.get("issuer", "Regulatory"),
                "circular_no": meta.get("circular_no", ""),
                "circular_date": meta.get("circular_date", ""),
                "status": meta.get("status", "ACTIVE"),
                "doc_title": meta.get("doc_title", ""),
                "source": meta.get("source", ""),
                "filename": meta.get("filename", ""),
                "is_statutory": meta.get("is_statutory", True)
            })
        else:
            # Split cleanly on sentences
            sentences = re.split(r"(?<=[.!?])\s+", body)
            buf = []
            curr_len = 0
            part = 1
            for sent in sentences:
                if curr_len + len(sent) > max_chunk_chars and buf:
                    chunk_text = " ".join(buf).strip()
                    c_label = f"{s['clause_no']} (Part {part})"
                    citation_str = (
                        f"{meta.get('issuer', 'Statutory')} {meta.get('doc_title', 'Circular')} "
                        f"({meta.get('circular_date', '2024')}), {c_label} [{meta.get('status', 'ACTIVE')}]"
                    )
                    final_chunks.append({
                        "text": f"[{c_label} - {s['header']}]\n{chunk_text}",
                        "clause_no": c_label,
                        "section_header": s["header"],
                        "citation": citation_str,
                        "issuer": meta.get("issuer", "Regulatory"),
                        "circular_no": meta.get("circular_no", ""),
                        "circular_date": meta.get("circular_date", ""),
                        "status": meta.get("status", "ACTIVE"),
                        "doc_title": meta.get("doc_title", ""),
                        "source": meta.get("source", ""),
                        "filename": meta.get("filename", ""),
                        "is_statutory": meta.get("is_statutory", True)
                    })
                    buf = [sent]
                    curr_len = len(sent)
                    part += 1
                else:
                    buf.append(sent)
                    curr_len += len(sent)
            if buf:
                chunk_text = " ".join(buf).strip()
                c_label = f"{s['clause_no']} (Part {part})" if part > 1 else s['clause_no']
                citation_str = (
                    f"{meta.get('issuer', 'Statutory')} {meta.get('doc_title', 'Circular')} "
                    f"({meta.get('circular_date', '2024')}), {c_label} [{meta.get('status', 'ACTIVE')}]"
                )
                final_chunks.append({
                    "text": f"[{c_label} - {s['header']}]\n{chunk_text}",
                    "clause_no": c_label,
                    "section_header": s["header"],
                    "citation": citation_str,
                    "issuer": meta.get("issuer", "Regulatory"),
                    "circular_no": meta.get("circular_no", ""),
                    "circular_date": meta.get("circular_date", ""),
                    "status": meta.get("status", "ACTIVE"),
                    "doc_title": meta.get("doc_title", ""),
                    "source": meta.get("source", ""),
                    "filename": meta.get("filename", ""),
                    "is_statutory": meta.get("is_statutory", True)
                })

    return final_chunks
