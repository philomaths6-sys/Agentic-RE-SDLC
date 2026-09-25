# finetune/prepare_dataset.py
"""Indian Banking & Financial Domain Synthetic Dataset Generator.
Synthesizes 350 high-fidelity, complex instruction-tuning JSONL pairs for:
1. data/training/agent_extraction.jsonl (Extraction & Classification Agent)
2. data/training/agent_sdlc.jsonl (SDLC Recommendation Agent)

Prioritizing Indian Statutory Authorities & Regulatory Directives:
- Reserve Bank of India (RBI)
- National Payments Corporation of India (NPCI)
- Unique Identification Authority of India (UIDAI)
- Digital Personal Data Protection Act, 2023 (DPDP Act)
- Prevention of Money Laundering Act (PMLA) & FIU-IND
- Securities and Exchange Board of India (SEBI)
"""

import os
import json
import random

os.makedirs("data/training", exist_ok=True)

INDIAN_BANKING_SCENARIOS = [
    # 1. UPI AutoPay & Recurring Mandates (NPCI / RBI)
    {
        "domain": "Retail Payments (UPI / NPCI)",
        "stakeholder_queries": [
            "We need to implement UPI AutoPay for mutual fund SIPs and loan EMIs. The user must authorize the mandate via UPI PIN, and pre-debit notifications must be dispatched via SMS/in-app 24 hours prior to every debit.",
            "For recurring OTT and insurance premium debits, our app needs NPCI UPI AutoPay support. All recurring transactions must notify the user 24 hours in advance and allow instant pause or revocation without merchant intervention.",
            "Integrate NPCI UPI 2.0 Recurring Mandates with automated pre-debit SMS alerts 24 hours before debit execution pursuant to RBI Digital Payment Security Controls."
        ],
        "req_id_prefix": "REQ-IN-UPI",
        "statement": "The payment gateway shall implement NPCI UPI AutoPay recurring mandate orchestration, enforcing UPI PIN authentication for mandate registration, automated pre-debit SMS notifications at least 24 hours prior to execution, and real-time mandate cancellation capabilities.",
        "categories": ["Retail Payments & Clearing (NPCI / RBI)", "Recurring Mandates", "Consumer Protection"],
        "priority": "Must Have",
        "priority_reason": "Mandatory compliance under RBI Circular on Processing of E-Mandate on Cards/UPI for Recurring Transactions (RBI/2019-20/47) & NPCI Operating Guidelines.",
        "regulatory_citations": [
            "RBI/2019-20/47 DPSS.CO.PD.No.447/02.14.003/2019-20",
            "NPCI UPI Procedural Guidelines on Recurring Mandate Execution"
        ],
        "acceptance_criteria": "Pre-debit alert dispatched >= 24 hours prior to debit timestamp; mandate modification/revocation reflected across sponsor bank switch within 500ms.",
        "dependencies": ["NPCI UPI Common Library (CL)", "Sponsor Bank Core Switch", "SMS Gateway (DLT Approved)"],
        "confidence_score": 0.98,
        "sdlc_model": "Agile-DevSecOps Hybrid",
        "sdlc_confidence": 95,
        "sdlc_justification": "High transaction throughput combined with strict NPCI certification cycles, real-time mandate state synchronization, and low-latency API contracts.",
        "selection_rationale": {
            "why_this_model": "Agile-DevSecOps Hybrid is chosen because NPCI UPI AutoPay requires iterative sprint delivery with continuous integration into NPCI simulator environments, real-time mandate state machine updates, and embedded security gates at every sprint boundary — all hallmarks of Agile combined with DevSecOps pipeline maturity.",
            "why_not_waterfall": "Waterfall is rejected because mandate APIs and pre-debit notification specs change frequently with NPCI circulars, making upfront-only requirements specification inadequate and costly to change mid-project.",
            "why_not_v_model": "Pure V-Model is rejected because NPCI certification is iterative and asynchronous — each phase generates feedback that requires sprint-based rework, which the strict sequential V-Model cannot accommodate without significant schedule risk.",
            "key_drivers": ["NPCI iterative certification cycles", "High-throughput mandate API contracts", "Continuous pre-debit notification pipeline", "Real-time mandate state synchronization", "Embedded SAST/DAST per sprint"]
        },
        "workflow": [
            {"phase": "Mandate Architecture & NPCI Alignment", "deliverable": "UPI AutoPay State Machine & DLT SMS Notification Blueprint", "security_gate": "NPCI Conformance Test Bed Pre-Review"},
            {"phase": "Switch Integration & Asynchronous Sprints", "deliverable": "High-Throughput Mandate Processing Daemon with mTLS", "security_gate": "SAST/DAST Zero Vulnerabilities in UPI PIN Entry Bridge"},
            {"phase": "NPCI Simulator & Stress Testing", "deliverable": "Pre-Debit Batch Simulation Report (5,000 mandates/sec)", "security_gate": "Zero mandate execution without prior 24h notification"},
            {"phase": "Sponsor Bank Certification & Go-Live", "deliverable": "NPCI Final Certificate of Operational Readiness", "security_gate": "Sponsor Bank CISO & Operations Sign-off"}
        ]
    },

    # 2. RBI Video-based Customer Identification Process (V-CIP) & Aadhaar Offline e-KYC
    {
        "domain": "Digital Onboarding & Identity",
        "stakeholder_queries": [
            "We are launching instant digital savings accounts. Customers must complete RBI-compliant Video KYC (V-CIP) with live facial liveness verification, geotagging within Indian territorial borders, and real-time NSDL PAN card verification.",
            "Our customer onboarding flow must implement RBI V-CIP: a trained bank official conducts a live audio-visual interaction, captures geotagged coordinates inside India, matches face against Aadhaar XML, and runs NSDL PAN validation.",
            "Build an automated V-CIP workflow verifying customer liveness, GPS geotagging confirmation on Indian soil, and instant CKYCR upload within 3 days of account opening."
        ],
        "req_id_prefix": "REQ-IN-VCIP",
        "statement": "The customer onboarding service shall execute Video-based Customer Identification Process (V-CIP) in accordance with RBI Master Direction, validating live facial interaction, Indian territorial GPS geotagging, real-time NSDL PAN verification, and automated CKYCR dossier upload within 3 days.",
        "categories": ["Customer Onboarding & Identity Governance (UIDAI / RBI)", "Biometric KYC", "Regulatory Compliance"],
        "priority": "Must Have",
        "priority_reason": "Statutory mandate under RBI Master Direction - Know Your Customer (KYC) Direction, 2016 (Section 18 V-CIP Guidelines).",
        "regulatory_citations": [
            "RBI/DBR/2015-16/18 Master Direction - Know Your Customer (KYC) Direction, 2016",
            "Prevention of Money-Laundering (Maintenance of Records) Rules, 2005"
        ],
        "acceptance_criteria": "System blocks V-CIP session if GPS coordinates fall outside India or proxy/VPN is detected; audio-visual recording encrypted and archived with timestamp and geotag.",
        "dependencies": ["Video Streaming WebRTC Server", "UIDAI Offline XML Engine", "NSDL / Protean PAN API", "CKYCR Clearing Registry"],
        "confidence_score": 0.97,
        "sdlc_model": "V-Model with DevSecOps Gates",
        "sdlc_confidence": 94,
        "sdlc_justification": "Strict regulatory accountability under PMLA and RBI guidelines where non-compliance directly risks bank licensing and legal sanctions.",
        "selection_rationale": {
            "why_this_model": "V-Model with DevSecOps Gates is chosen because RBI V-CIP has a rigid, legally defined procedural specification where every development phase must have a corresponding verification artifact — this maps precisely to V-Model's phase-verification contract. DevSecOps gates are embedded to enforce CERT-In security validation at each milestone.",
            "why_not_agile": "Pure Agile is rejected because the RBI KYC Master Direction does not permit iterative requirement changes to the customer identification workflow — the process is legally fixed, and any sprint-based deviation could constitute non-compliance.",
            "why_not_waterfall": "Plain Waterfall is rejected because modern biometric liveness and deepfake detection technologies require continuous security patch integration, which pure Waterfall cannot accommodate without a formal DevSecOps overlay.",
            "key_drivers": ["RBI V-CIP legally fixed procedural spec", "PMLA criminal liability for KYC failures", "CERT-In mandatory security audit", "CKYCR registry synchronization SLA", "Biometric liveness verification hardening"]
        },
        "workflow": [
            {"phase": "Regulatory Process Spec & Legal Sign-off", "deliverable": "V-CIP Standard Operating Procedure (SOP) & Security Blueprint", "security_gate": "Principal Officer & Chief Compliance Officer Approval"},
            {"phase": "Geofencing & Video Cryptographic Hardening", "deliverable": "Tamper-Evident Geotagged Video Recording Pipeline", "security_gate": "Deepfake & Virtual Camera Injection Attack Pen-Test"},
            {"phase": "CERT-In Empaneled Security Audit", "deliverable": "Independent Third-Party Application Security Audit Report", "security_gate": "Zero Unmitigated Medium/High/Critical Findings"},
            {"phase": "Production Commissioning & CKYCR Sync", "deliverable": "Automated Central KYC Registry Batch Verification", "security_gate": "Internal Audit & Concurrent Auditor Formal Sign-off"}
        ]
    },

    # 3. UIDAI Aadhaar Data Vault (ADV) & Number Masking
    {
        "domain": "Identity Data Security",
        "stakeholder_queries": [
            "UIDAI mandates that we cannot store 12-digit Aadhaar numbers in any database. We must implement an isolated Aadhaar Data Vault with HSM encryption and replace all Aadhaar numbers with random reference keys.",
            "All customer Aadhaar numbers must be masked ('XXXX-XXXX-1234') in application logs, customer receipts, and database tables, with only the reference key stored in core tables per UIDAI circulars.",
            "We need an Aadhaar Data Vault (ADV) implementation using FIPS 140-2 HSM for tokenization, ensuring raw Aadhaar is stored only in an isolated encrypted vault."
        ],
        "req_id_prefix": "REQ-IN-ADV",
        "statement": "The core identity store shall establish an isolated Aadhaar Data Vault (ADV) utilizing FIPS 140-2 Level 3 Hardware Security Modules to tokenize 12-digit Aadhaar numbers into non-reversible reference keys, mandating masking of the first eight digits across all downstream systems and logs.",
        "categories": ["Identity Security (UIDAI)", "Aadhaar Data Vault", "Cryptography"],
        "priority": "Must Have",
        "priority_reason": "Statutory obligation under UIDAI Circular on Aadhaar Data Vault (Circular No. 11020/205/2017) and Aadhaar Act 2016.",
        "regulatory_citations": [
            "UIDAI Circular No. 11020/205/2017 - Implementation of Aadhaar Data Vault",
            "Aadhaar (Targeted Delivery of Financial and Other Subsidies, Benefits and Services) Act, 2016"
        ],
        "acceptance_criteria": "Zero raw Aadhaar numbers found in non-ADV databases, logs, or caches; ADV access strictly restricted to dual-authorized services via mTLS.",
        "dependencies": ["FIPS 140-2 Level 3 HSM", "Aadhaar Vault Service", "Database Vault Token Table"],
        "confidence_score": 0.99,
        "sdlc_model": "V-Model with DevSecOps Gates",
        "sdlc_confidence": 96,
        "sdlc_justification": "Severe criminal liability and statutory audit under the Aadhaar Act requiring formal cryptographic validation and network isolation.",
        "selection_rationale": {
            "why_this_model": "V-Model with DevSecOps Gates is chosen because UIDAI's Aadhaar Data Vault specification is a fixed technical standard — each implementation step must be independently verified via CERT-In audit before the next phase. The V-Model's strict phase-verification pairing is the only methodology that ensures no vault is commissioned without formal cryptographic attestation.",
            "why_not_agile": "Agile is rejected because Aadhaar tokenization is a one-time architectural migration with zero tolerance for incremental rollout — any partial Aadhaar vault state exposes the organization to criminal liability under the Aadhaar Act Section 29.",
            "why_not_waterfall": "Plain Waterfall is rejected because HSM firmware updates and cryptographic key ceremonies require continuous DevSecOps pipeline integration for automated regression testing of the vault's security posture.",
            "key_drivers": ["UIDAI criminal liability under Aadhaar Act", "FIPS 140-2 Level 3 HSM certification requirement", "CERT-In mandatory pre-go-live audit", "Zero-plaintext residue in all databases", "Isolated network vault architecture"]
        },
        "workflow": [
            {"phase": "Cryptographic Architecture & Vault Isolation", "deliverable": "Aadhaar Data Vault Network Segmentation & HSM Key Topology", "security_gate": "UIDAI Compliance Audit Pre-Assessment"},
            {"phase": "Tokenization Engine & Redaction Filter Build", "deliverable": "Aadhaar Reference Key Tokenizer & Regex Redaction Daemon", "security_gate": "Static Code & Memory Dump Zero-Plaintext Audit"},
            {"phase": "Automated Database & Log Inspection", "deliverable": "Automated Hex/String Database Scan across all 300+ tables", "security_gate": "Zero unmasked 12-digit numeric sequences detected"},
            {"phase": "CERT-In Audit & Final Commissioning", "deliverable": "CERT-In Empaneled Auditor Aadhaar Vault Compliance Certificate", "security_gate": "CISO & Bank Legal Counsel Attestation"}
        ]
    },

    # 4. RBI Sovereign Data Localization (Payment System Data Storage 2018)
    {
        "domain": "Data Residency & Sovereignty",
        "stakeholder_queries": [
            "Under RBI directive on Storage of Payment System Data, all transaction logs, authorization tokens, customer card details, and settlement records must reside exclusively on servers physically located in India.",
            "Our cloud deployment on AWS/Azure must be restricted strictly to Mumbai/Hyderabad regions (ap-south-1/2) with zero cross-border replication of payment data to comply with RBI data localization.",
            "Ensure complete compliance with RBI 2018 Data Localization circular: no domestic payment processing data can be stored outside Indian soil, and international transactions processed abroad must be deleted from foreign servers within 24 hours."
        ],
        "req_id_prefix": "REQ-IN-LOC",
        "statement": "The cloud infrastructure and database tier shall enforce sovereign data residency by hosting all end-to-end payment transaction data, settlement ledgers, and audit logs exclusively on physical infrastructure located within the Republic of India, with zero foreign persistent replication.",
        "categories": ["Sovereign Data Residency & Localization (RBI 2018 Directive)", "Infrastructure Security", "Cloud Governance"],
        "priority": "Must Have",
        "priority_reason": "Non-negotiable statutory directive under RBI DPSS.CO.OD No. 2785/06.08.005/2017-18.",
        "regulatory_citations": [
            "RBI/2017-18/153 DPSS.CO.OD No. 2785/06.08.005/2017-18 (Storage of Payment System Data)",
            "RBI Clarifications on Storage of Payment System Data (June 2019)"
        ],
        "acceptance_criteria": "All persistent volumes and replication pairs verified to reside strictly in Indian cloud availability zones; annual System Audit Report (SAR) by CERT-In auditor certified clean.",
        "dependencies": ["India-Region Cloud Infrastructure (AWS ap-south / Azure India)", "Database Replication Topology", "Network Egress Gateway"],
        "confidence_score": 0.98,
        "sdlc_model": "Agile-DevSecOps Hybrid",
        "sdlc_confidence": 94,
        "sdlc_justification": "Infrastructure-as-Code policy enforcement coupled with automated CI/CD region locking and continuous compliance monitoring.",
        "selection_rationale": {
            "why_this_model": "Agile-DevSecOps Hybrid is chosen because cloud data localization requires continuous Infrastructure-as-Code (IaC) policy drift detection, which is best managed via iterative sprint cycles with automated CI/CD guardrails that block cross-border region deployments. Each sprint yields testable cloud sovereignty assertions.",
            "why_not_v_model": "V-Model is rejected because cloud infrastructure evolves rapidly — new availability zones, egress routes, and cloud services must be evaluated iteratively, not fixed in a single upfront specification.",
            "why_not_waterfall": "Waterfall is rejected because annual CERT-In System Audit Report requirements demand continuous compliance monitoring dashboards, which cannot be built as a one-shot deliverable but require sprint-by-sprint observability tooling.",
            "key_drivers": ["Terraform IaC region policy enforcement", "Automated egress audit per CI/CD pipeline", "RBI annual System Audit Report (SAR) compliance", "Multi-cloud Indian region governance", "Zero foreign persistent replication guarantee"]
        },
        "workflow": [
            {"phase": "Cloud Sovereignty Architecture & IaC Guardrails", "deliverable": "Terraform Policy Guardrails Restricting Regions to India Only", "security_gate": "Cloud Architecture Review Board Sign-off"},
            {"phase": "Egress Traffic & Data Flow Hardening", "deliverable": "Network Firewall Rules Blocking Foreign Database Replicas", "security_gate": "Automated Cloud Egress Audit: Zero foreign data egress"},
            {"phase": "System Audit Report (SAR) by CERT-In", "deliverable": "Independent CERT-In Empaneled System Audit Report", "security_gate": "Audit confirms 100% data on-soil residency"},
            {"phase": "Annual RBI Filing & Regulatory Attestation", "deliverable": "Executive Compliance Filing to Department of Payment & Settlement Systems (DPSS)", "security_gate": "Board of Directors Compliance Review"}
        ]
    },

    # 5. RBI Digital Lending Guidelines 2022 & Default Loss Guarantee (DLG)
    {
        "domain": "Digital Lending & FinTech",
        "stakeholder_queries": [
            "To comply with RBI Digital Lending Guidelines, loan disbursals must flow directly from our bank account to the borrower's bank account. Fintech Lending Service Provider (LSP) pass-through or pool accounts are strictly prohibited.",
            "Our digital loan journey must present an automated Key Fact Statement (KFS) detailing the Annual Percentage Rate (APR), processing fees, and a 3-day cooling-off period before loan agreement execution.",
            "Implement RBI 2022 Digital Lending controls: direct bank-to-borrower fund transfer, automated KFS generation with all-inclusive APR, and no biometric/camera permissions stored on LSP mobile apps."
        ],
        "req_id_prefix": "REQ-IN-DLG",
        "statement": "The digital lending platform shall mandate direct fund disbursement from the Regulated Entity (RE) escrow to the borrower's verified bank account without intermediary LSP pool accounts, and generate a standardized Key Fact Statement (KFS) with Annual Percentage Rate (APR) and 3-day cooling-off period.",
        "categories": ["Digital Lending & Credit Architecture (RBI 2022 Guidelines)", "Fintech Governance", "Consumer Protection"],
        "priority": "Must Have",
        "priority_reason": "Statutory mandate under RBI Guidelines on Digital Lending (RBI/2022-23/111) & Default Loss Guarantee (DLG) Framework.",
        "regulatory_citations": [
            "RBI/2022-23/111 DOR.CRE.REC.66/21.07.001/2022-23 (Guidelines on Digital Lending)",
            "RBI/2023-24/41 DOR.CRE.REC.21/21.07.001/2023-24 (Default Loss Guarantee in Digital Lending)"
        ],
        "acceptance_criteria": "Zero funds routed through third-party LSP escrow; KFS accepted via digital signature/OTP with cooling-off period recorded in audit ledger.",
        "dependencies": ["Core Banking Loan Disbursal Switch", "KFS PDF Rendering Engine", "Borrower Bank Account Penny Drop Verification"],
        "confidence_score": 0.97,
        "sdlc_model": "V-Model with DevSecOps Gates",
        "sdlc_confidence": 93,
        "sdlc_justification": "Rigid legal and regulatory framework where violations lead to immediate cancellation of digital lending partnerships and regulatory fines.",
        "selection_rationale": {
            "why_this_model": "V-Model with DevSecOps Gates is chosen because RBI Digital Lending Guidelines 2022 define a legally fixed workflow (KFS generation → borrower consent → direct disbursal) where each phase must be independently verified by compliance and legal teams before the next is executed — a direct match for V-Model's phase-gating structure.",
            "why_not_agile": "Agile is rejected because the RBI's prohibition on LSP pool accounts and the KFS APR calculation formula are legally mandated and cannot be iteratively refined — any sprint-based deviation risks regulatory sanction.",
            "why_not_waterfall": "Plain Waterfall is rejected because app store security reviews (Google Play / Apple) and mobile SDK permission audits require continuous DevSecOps integration, not a one-shot build.",
            "key_drivers": ["RBI direct disbursal mandate (no LSP escrow)", "Legally fixed KFS template and APR formula", "3-day cooling-off period audit trail", "DLG 5% portfolio cap monitoring", "Mobile SDK permission compliance"]
        },
        "workflow": [
            {"phase": "Product Compliance & KFS Legal Engineering", "deliverable": "Standardized KFS Template & All-inclusive APR Computation Formula", "security_gate": "Chief Risk Officer & Head of Retail Lending Approval"},
            {"phase": "Direct Disbursal & Penny-Drop Pipeline Build", "deliverable": "Account Verification Microservice & Direct NEFT/IMPS Disbursal Switch", "security_gate": "Ledger Audit: 0 pass-through transactions"},
            {"phase": "App Permission & Data Privacy Audit", "deliverable": "Mobile SDK Permission Audit (Zero contact list/file access)", "security_gate": "Google Play / Apple App Store Banking Security Review"},
            {"phase": "Regulatory Filing & Portfolio Monitoring", "deliverable": "DLG 5% Cap Verification Dashboard", "security_gate": "Concurrent Auditor Quarterly Sign-off"}
        ]
    },

    # 6. RBI Card-on-File Tokenization (CoFT)
    {
        "domain": "Card Payments & Tokenization",
        "stakeholder_queries": [
            "Under RBI Card-on-File Tokenization (CoFT) rules, our e-commerce platform and merchant partners can no longer store 16-digit card numbers. We must integrate RuPay and card network tokenization with device binding.",
            "Replace all stored customer debit/credit cards with CoFT tokens issued by card networks. Merchants can only store the last 4 digits and card issuer name for display purposes.",
            "Implement RBI Card-on-File Tokenization with explicit customer consent, two-factor authentication for token creation, and merchant token management portals."
        ],
        "req_id_prefix": "REQ-IN-COFT",
        "statement": "The card acquiring switch shall implement Card-on-File Tokenization (CoFT) in conformance with RBI directives, replacing stored Primary Account Numbers with network-issued tokens and restricting merchant storage to the last four digits and card issuer name.",
        "categories": ["Retail Payments & Clearing (NPCI / RBI)", "Card-on-File Tokenization", "Cybersecurity"],
        "priority": "Must Have",
        "priority_reason": "Mandatory directive under RBI Circular on Tokenisation – Card Transactions (RBI/2021-22/96).",
        "regulatory_citations": [
            "RBI/2021-22/96 DPSS.CO.PD No.S-516/02-14-003/2021-22",
            "NPCI RuPay Tokenization Framework Guidelines"
        ],
        "acceptance_criteria": "All legacy card databases purged of plaintext PANs; token requests authenticated via AFA (Additional Factor of Authentication).",
        "dependencies": ["NPCI RuPay Token Service (RTS)", "Card Network Token Connectors (Visa VTS / Mastercard MDES)", "Merchant Token Vault"],
        "confidence_score": 0.98,
        "sdlc_model": "Agile-DevSecOps Hybrid",
        "sdlc_confidence": 95,
        "sdlc_justification": "Ecosystem integration across card networks, payment aggregators, and merchants requiring rigorous integration testing and zero downtime.",
        "selection_rationale": {
            "why_this_model": "Agile-DevSecOps Hybrid is chosen because Card-on-File Tokenization spans multiple external ecosystems (Visa VTS, Mastercard MDES, NPCI RuPay RTS, and merchant payment aggregators), each with independent test certification cycles. Agile sprints allow iterative network-by-network integration while DevSecOps gates enforce automated PAN-zeroization verification at every build.",
            "why_not_v_model": "V-Model is rejected because merchant tokenization scope expands over time as new merchants onboard — the system must accommodate continuous token registration workflows that a fixed V-Model specification cannot easily handle.",
            "why_not_waterfall": "Waterfall is rejected because card network certification timelines (Visa, Mastercard, NPCI) are asynchronous and external — waiting for all certifications before building the next phase would delay the entire program unacceptably.",
            "key_drivers": ["Multi-network token certification (Visa/Mastercard/RuPay)", "Zero-downtime PAN database zeroization", "Merchant AFA consent pipeline", "QSA PCI-DSS attestation requirement", "Legacy card database migration sprints"]
        },
        "workflow": [
            {"phase": "Token Protocol Mapping & Network Integration", "deliverable": "CoFT Token Lifecycle Architecture & Cryptographic Key Spec", "security_gate": "Card Network Conformance Simulator Test Passed"},
            {"phase": "Legacy PAN Purge & Token Migration", "deliverable": "Database Zeroization Script & Token Mapping Table", "security_gate": "Automated Database Scan: 100% eradication of raw card PANs"},
            {"phase": "AFA Consent & Interoperability Testing", "deliverable": "Merchant Checkout Tokenization Test Matrix", "security_gate": "Zero-friction token payment latency < 800ms"},
            {"phase": "Production Cutover & QSA Verification", "deliverable": "CoFT Attestation Report for RBI", "security_gate": "Payment Aggregator CISO Sign-off"}
        ]
    },

    # 7. RBI Account Aggregator (AA) Ecosystem (Sahamati / ReBIT)
    {
        "domain": "Open Finance & Data Aggregation",
        "stakeholder_queries": [
            "Our bank needs to act as a Financial Information Provider (FIP) in the RBI Account Aggregator ecosystem, sharing bank statements with NBFCs using ReBIT XML/JSON schemas and Sahamati consent handles.",
            "Implement Account Aggregator (AA) consent framework: users grant time-bound consent on Sahamati-certified AAs to share transaction history with lenders, encrypted with the lender's public key.",
            "Build Financial Information Provider (FIP) APIs following ReBIT technical standards with digital signature verification on all Account Aggregator data payloads."
        ],
        "req_id_prefix": "REQ-IN-AA",
        "statement": "The bank API gateway shall implement the RBI Account Aggregator framework as a Financial Information Provider (FIP), delivering end-to-end encrypted financial data to authorized Financial Information Users (FIUs) via ReBIT standard schemas and Sahamati consent artifacts.",
        "categories": ["Open Finance (RBI Account Aggregator)", "Consent Architecture", "API Security"],
        "priority": "Must Have",
        "priority_reason": "RBI Master Direction on Non-Banking Financial Company - Account Aggregator (Reserve Bank) Directions, 2016.",
        "regulatory_citations": [
            "RBI/DNBR/2016-17/45 Master Direction- Non-Banking Financial Company - Account Aggregator (Reserve Bank) Directions, 2016",
            "ReBIT Technical Specifications for Account Aggregators (Data Flows v1.1.2)"
        ],
        "acceptance_criteria": "Data encrypted end-to-end using ephemeral ECDH key pairs; FIP decrypts nothing in transit; data sharing immediately ceases upon consent revocation.",
        "dependencies": ["ReBIT Standard Data Adapter", "Sahamati Central Registry", "HSM Digital Signing Gateway"],
        "confidence_score": 0.96,
        "sdlc_model": "Agile-DevSecOps Hybrid",
        "sdlc_confidence": 92,
        "sdlc_justification": "Ecosystem API interface demanding protocol conformance, asynchronous polling architectures, and cryptographic payload verification.",
        "selection_rationale": {
            "why_this_model": "Agile-DevSecOps Hybrid is chosen because the RBI Account Aggregator ecosystem involves iterative API conformance testing against multiple certified Account Aggregators (Finvu, PhonePe, CAMS FinServ, etc.), each releasing updated ReBIT schema versions. Agile allows rapid schema adaptation while DevSecOps gates enforce ECDH key agreement verification at every sprint.",
            "why_not_v_model": "V-Model is rejected because ReBIT Technical Specifications are living documents updated by IBA/RBI committees — the V-Model's fixed specification phase would be obsolete before the first implementation sprint is complete.",
            "why_not_waterfall": "Waterfall is rejected because the FIP data consent lifecycle involves asynchronous callback events from multiple AAs, which require continuous integration testing that a sequential waterfall delivery cannot support.",
            "key_drivers": ["ReBIT schema version evolution", "Sahamati multi-AA conformance certification", "ECDH cryptographic payload verification", "Consent revocation real-time propagation", "Account Aggregator sandbox certification"]
        },
        "workflow": [
            {"phase": "ReBIT Schema Engineering & Consent Matrix", "deliverable": "FIP Data Connector Architecture & ECDH Key Agreement Spec", "security_gate": "Sahamati Interoperability Conformance Testing"},
            {"phase": "Asynchronous Data Fetch & Signing Build", "deliverable": "Signed Financial Information Microservice (ReBIT Format)", "security_gate": "Cryptographic Verification: Payload tamper test"},
            {"phase": "AA Sandbox Certification", "deliverable": "End-to-End AA Test Report across 4 Certified Account Aggregators", "security_gate": "Zero data leakage on expired consent trial"},
            {"phase": "Production Go-Live", "deliverable": "Account Aggregator Operational Monitoring Dashboard", "security_gate": "Internal Compliance & IT Steering Committee Approval"}
        ]
    },

    # 8. Prevention of Money Laundering Act (PMLA) & FIU-IND Reporting
    {
        "domain": "Financial Intelligence & AML",
        "stakeholder_queries": [
            "Our transaction monitoring platform must automatically identify suspicious transaction patterns (structuring below ₹10 Lakhs, sudden turnover surges) and compile an XML Suspicious Transaction Report (STR) for FIU-IND within 7 days.",
            "To comply with PMLA 2002 and RBI guidelines, all cash transactions exceeding ₹10 Lakhs in a month must be compiled into monthly Cash Transaction Reports (CTR) and filed on the FINnet 2.0 gateway.",
            "Implement automated AML red-flag rules and Suspicious Transaction Report (STR) batch exporter conforming to FIU-IND FINnet 2.0 XML specifications."
        ],
        "req_id_prefix": "REQ-IN-AML",
        "statement": "The AML surveillance engine shall detect suspicious transaction typologies under the Prevention of Money Laundering Act (PMLA) and automatically generate validated FINnet 2.0 XML dossiers for Suspicious Transaction Reports (STR) within seven working days of suspicion.",
        "categories": ["AML & Financial Intelligence Surveillance (PMLA / FIU-IND)", "Regulatory Reporting", "Transaction Monitoring"],
        "priority": "Must Have",
        "priority_reason": "Statutory obligation under Prevention of Money Laundering Act, 2002 (PMLA § 12) & Financial Intelligence Unit - India (FIU-IND) Guidelines.",
        "regulatory_citations": [
            "Prevention of Money Laundering Act, 2002 (Act No. 15 of 2003) § 12",
            "FIU-IND FINnet 2.0 Reporting Format Guidelines"
        ],
        "acceptance_criteria": "STR generation workflow triggers within 24h of confirmed red-flag; XML dossier validates 100% against FIU-IND FINnet 2.0 XSD schemas.",
        "dependencies": ["Transaction Surveillance Engine", "Customer Risk Profiler", "FINnet 2.0 XML Batch Generator"],
        "confidence_score": 0.98,
        "sdlc_model": "V-Model",
        "sdlc_confidence": 94,
        "sdlc_justification": "Statutory legal filings with severe criminal and regulatory penalties for non-filing, demanding deterministic test suites and strict confidentiality controls.",
        "selection_rationale": {
            "why_this_model": "V-Model is chosen because PMLA STR filing is a legal obligation with deterministic requirements — the FIU-IND FINnet 2.0 XML schema is fixed by statute, and each development phase (red flag rule definition → surveillance engine → FINnet XML generation → test bed submission) must be formally verified before proceeding. Criminal penalties for missed STRs demand zero-defect verification at every phase.",
            "why_not_agile": "Agile is rejected because PMLA red flag typologies are legally defined and cannot be iteratively revised without legal counsel sign-off at every change — sprint-based requirement volatility creates unacceptable criminal exposure.",
            "why_not_agile_devsecops": "Agile-DevSecOps Hybrid is rejected because the FINnet 2.0 XML schema does not change iteratively — the entire pipeline is specified by FIU-IND regulation, making sprint-based iteration wasteful overhead without regulatory benefit.",
            "key_drivers": ["PMLA Section 12 criminal liability", "FINnet 2.0 XSD schema fixed by statute", "Principal Officer mandatory sign-off at each phase", "Deterministic STR within 7-working-day SLA", "Strict confidentiality controls for STR data"]
        },
        "workflow": [
            {"phase": "Typology Modeling & FINnet 2.0 Schema Spec", "deliverable": "PMLA Red Flag Rule Matrix & FINnet 2.0 XML Mapping", "security_gate": "Principal Officer (AML) & Legal Counsel Sign-off"},
            {"phase": "Surveillance Engine & Batch Builder", "deliverable": "Real-Time Structuring & Velocity Monitoring Engine", "security_gate": "Backtest: Zero missed statutory cash threshold anomalies"},
            {"phase": "FIU-IND Test Bed Conformance", "deliverable": "Test Filing Acknowledgement from FINnet 2.0 Test Bed", "security_gate": "Zero XML validation schema warnings or errors"},
            {"phase": "Operational Commissioning", "deliverable": "Confidential STR Handling & Access Control Policy", "security_gate": "CISO & Audit Committee Formal Sign-off"}
        ]
    },

    # 9. Digital Personal Data Protection Act, 2023 (DPDP Act)
    {
        "domain": "Data Privacy & Governance",
        "stakeholder_queries": [
            "Under the Digital Personal Data Protection Act 2023 (DPDP Act), all mobile banking consent requests must be presented in clear, plain language in English and all 22 Eighth Schedule languages, with an itemized notice of personal data collected.",
            "We must implement a Data Principal Grievance Redressal pipeline resolving customer data privacy disputes within 72 hours, with automated data erasure once processing purpose is fulfilled.",
            "Build a DPDP Act compliant consent architecture with multi-lingual notices, explicit consent logging, and instant data erasure workflows."
        ],
        "req_id_prefix": "REQ-IN-DPDP",
        "statement": "The privacy and consent architecture shall enforce the Digital Personal Data Protection Act, 2023, presenting clear statutory notices in English and 22 Eighth Schedule Indian languages, maintaining immutable consent receipts, and enabling Data Principal rights for grievance redressal and data erasure.",
        "categories": ["Digital Personal Data Protection & Privacy (DPDP Act 2023)", "Consent Management", "Data Governance"],
        "priority": "Must Have",
        "priority_reason": "Statutory liability under Digital Personal Data Protection Act, 2023 (Sections 6, 8, and 12) with regulatory penalties up to ₹250 Crores.",
        "regulatory_citations": [
            "Digital Personal Data Protection Act, 2023 (Act No. 22 of 2023)",
            "Ministry of Electronics and Information Technology (MeitY) DPDP Rules"
        ],
        "acceptance_criteria": "Consent captured with timestamp, IP, and specific purpose code; consent withdrawal immediately halts all non-statutory data processing within 60 seconds.",
        "dependencies": ["Multi-lingual Consent UI Engine", "Consent Receipt Repository", "Data Principal Grievance Portal"],
        "confidence_score": 0.97,
        "sdlc_model": "Agile-DevSecOps Hybrid",
        "sdlc_confidence": 93,
        "sdlc_justification": "Evolving regulatory notices coupled with multi-service data discovery and enterprise-wide consent enforcement pipelines.",
        "selection_rationale": {
            "why_this_model": "Agile-DevSecOps Hybrid is chosen because the DPDP Act 2023 rules and MeitY guidelines are still being finalized — the consent architecture must be built iteratively to accommodate rule amendments. Each Agile sprint delivers a verifiable consent sub-system (notice rendering → consent capture → purpose binding → erasure pipeline), with DevSecOps gates enforcing DPIA at each milestone.",
            "why_not_v_model": "V-Model is rejected because MeitY DPDP Rules are published in phases — building a fixed specification upfront would require complete re-specification with each rule gazette notification, making V-Model economically non-viable.",
            "why_not_waterfall": "Waterfall is rejected because the 22-language notice matrix requires continuous localization testing and user feedback integration, which demands iterative sprint cycles rather than a single build-and-deploy cycle.",
            "key_drivers": ["DPDP Act evolving regulatory rules", "22 Eighth Schedule language localization", "Enterprise-wide consent propagation pipeline", "Real-time erasure within 60 seconds of revocation", "DPIA mandatory at each architecture milestone"]
        },
        "workflow": [
            {"phase": "DPDP Legal Interpretation & Consent Taxonomy", "deliverable": "22-Language Statutory Notice Matrix & Purpose Specification", "security_gate": "Data Protection Officer (DPO) & Legal Sign-off"},
            {"phase": "Consent Architecture & Revocation Microservice", "deliverable": "Consent Management Platform with Webhook Propagation", "security_gate": "Downstream Pipeline Test: Zero processing after consent withdrawal"},
            {"phase": "Data Erasure & Anonymization Audit", "deliverable": "Automated Erasure Verification Report across Data Lakes", "security_gate": "DPIA (Data Protection Impact Assessment) Formal Approval"},
            {"phase": "Grievance Pipeline Go-Live", "deliverable": "72-Hour Grievance SLA Dashboard & Escalation Runbook", "security_gate": "Executive Risk Committee Sign-off"}
        ]
    },

    # 10. SEBI Cyber Security and Cyber Resilience Framework (CSCRF)
    {
        "domain": "Capital Markets & Brokerage",
        "stakeholder_queries": [
            "SEBI mandates mandatory Two-Factor Authentication (2FA) involving biometric or TOTP for all stock trading and Demat account logins. Password-only login is strictly prohibited.",
            "Our online broking platform must comply with SEBI's Cyber Security and Cyber Resilience Framework (CSCRF), enforcing API rate-limiting, encrypted order routing, and immutable audit logs of algorithmic trading strategies.",
            "Implement SEBI CSCRF guidelines for our trading app: biometric 2FA for web/mobile, encrypted FIX protocol order messages, and real-time anomaly detection for unauthorized account access."
        ],
        "req_id_prefix": "REQ-IN-SEBI",
        "statement": "The securities trading and demat portal shall enforce the SEBI Cyber Security and Cyber Resilience Framework (CSCRF), mandating two-factor authentication (TOTP/Biometric) for all client logins, end-to-end cryptographic order signing, and automated trading anomaly detection.",
        "categories": ["Capital Markets (SEBI CSCRF)", "Trading Security", "Two-Factor Authentication"],
        "priority": "Must Have",
        "priority_reason": "Statutory mandate under SEBI Master Circular on Cyber Security and Cyber Resilience Framework (CSCRF).",
        "regulatory_citations": [
            "SEBI/HO/MIRSD/MIRSD-PoD-1/P/CIR/2023/147 Master Circular on Cyber Security and Cyber Resilience",
            "SEBI Circular on Two-Factor Authentication for Trading Accounts"
        ],
        "acceptance_criteria": "Single-factor password login completely disabled; all order payloads digitally hashed and timestamped with microsecond precision for exchange audit trail.",
        "dependencies": ["TOTP / Biometric Auth Gateway", "FIX Engine with TLS 1.3", "Exchange Order Logging Vault"],
        "confidence_score": 0.98,
        "sdlc_model": "V-Model with DevSecOps Gates",
        "sdlc_confidence": 95,
        "sdlc_justification": "High-frequency, mission-critical market infrastructure with stringent zero-trust access rules and SEBI regulatory inspection audits.",
        "selection_rationale": {
            "why_this_model": "V-Model with DevSecOps Gates is chosen because SEBI CSCRF defines a legally mandated cybersecurity architecture specification — every access control, encryption standard, and audit log format is prescribed by the master circular. V-Model enforces formal verification at each phase (security architecture → 2FA implementation → FIX protocol encryption → audit trail), while DevSecOps gates embed continuous VAPT cycles between phases.",
            "why_not_agile": "Agile is rejected because SEBI CSCRF does not permit iterative security control deployment — all mandatory controls must be implemented and verified together before market operations commence to avoid exchange regulatory penalties.",
            "why_not_waterfall": "Plain Waterfall is rejected because zero-day vulnerabilities in trading infrastructure require continuous patch management via DevSecOps pipelines — a waterfall release cycle would leave the platform exposed between annual releases.",
            "key_drivers": ["SEBI CSCRF legally mandated security spec", "Zero-trust 2FA for all trading sessions", "Microsecond precision order audit trail", "Annual VAPT and SEBI compliance filing", "FIX protocol TLS 1.3 encrypted order routing"]
        },
        "workflow": [
            {"phase": "CSCRF Security Architecture Review", "deliverable": "Trading Perimeter Threat Model & Zero-Trust Access Blueprint", "security_gate": "SEBI-Empaneled Cybersecurity Auditor Sign-off"},
            {"phase": "2FA & High-Speed FIX Integration", "deliverable": "Ultra-Low Latency Auth & Order Routing Microservice", "security_gate": "Pen-Test: Zero bypass on 2FA session token hijacking"},
            {"phase": "Algorithmic Order Audit Trail Testing", "deliverable": "Microsecond Precision Order Audit Log Verification Suite", "security_gate": "Exchange Mock Trading Drill Conformance"},
            {"phase": "SEBI Compliance Filing", "deliverable": "Annual Cyber Security Audit Report to SEBI and Stock Exchanges", "security_gate": "Board IT Strategy Committee Formal Approval"}
        ]
    },

    # 11. Conversational Banking Chatbot (Prototyping Model)
    {
        "domain": "Digital Banking UX & Conversational AI",
        "stakeholder_queries": [
            "We want to build a conversational AI banking chatbot that helps customers check balances, initiate UPI transfers, and raise service requests. The UX and dialogue flows are completely unclear — we need user feedback before full development.",
            "Our bank wants to launch a WhatsApp-based banking assistant. We have no idea what flows customers prefer, what language they use, or how they phrase banking queries. We need to prototype and test multiple dialogue flows with real users before building.",
            "Build a multi-lingual AI banking assistant for rural customers (Hindi, Tamil, Telugu). The exact user interaction patterns, query types, and language preferences are unknown and must be discovered through iterative prototype testing."
        ],
        "req_id_prefix": "REQ-IN-CHATBOT",
        "statement": "The conversational banking assistant shall support UPI transfers, balance inquiry, and FD booking through a natural language interface, with user experience flows validated through iterative prototype testing with actual customer segments before production build.",
        "categories": ["Conversational Banking & AI UX", "Digital Channels", "Customer Experience"],
        "priority": "Should Have",
        "priority_reason": "RBI Guidelines on Internet Banking and Mobile Banking Services mandate user interface clarity and accessibility for digital channels.",
        "regulatory_citations": [
            "RBI/2011-12/379 Guidelines on Internet Banking Facilities for Customers",
            "NPCI UPI Product Guidelines on Third-Party Application Providers"
        ],
        "acceptance_criteria": "Prototype tested with minimum 50 real bank customers across 3 iteration cycles; final dialogue flows achieve >85% task completion rate before development freeze.",
        "dependencies": ["NLP/LLM Dialogue Engine", "UPI API Integration", "WhatsApp Business API"],
        "confidence_score": 0.90,
        "sdlc_model": "Prototyping Model",
        "sdlc_confidence": 90,
        "sdlc_justification": "Conversational banking UX requirements are inherently exploratory — customer dialogue patterns, preferred languages, and query phrasing are unknown until tested with real users. Prototyping enables rapid UI/dialogue iteration based on direct customer feedback before committing to full development costs.",
        "selection_rationale": {
            "why_this_model": "Prototyping Model is chosen because the conversational banking assistant has highly uncertain UX requirements — no one knows upfront how rural customers phrase banking queries in Hindi or Tamil, which flows they abandon, or what terminology they understand. Building throwaway prototypes and testing with 50 real users per iteration cycle is the only way to discover the correct dialogue design before freezing the production specification.",
            "why_not_alternative": "Agile-DevSecOps Hybrid is rejected because sprint-based development assumes requirements are sufficiently understood to write user stories — but for a first-ever multi-lingual conversational banking product, the requirements are fundamentally unknown until user research is done. Building production code on unknown UX would result in costly rework.",
            "key_drivers": ["Unknown customer dialogue patterns", "Multi-lingual UX validation required", "High UX rework cost without prototyping", "Rural digital literacy variance", "Iterative user testing before specification freeze"]
        },
        "workflow": [
            {"phase": "Exploratory Prototype 1 (Text UI Mock)", "deliverable": "Low-Fidelity Chatbot Dialogue Flow Mock", "security_gate": "No real customer PII in prototype; test with synthetic data"},
            {"phase": "User Testing Round 1 (50 customers)", "deliverable": "User Research Report: Task completion, Drop-off points, Language gaps", "security_gate": "Data Principal consent obtained for research per DPDP Act"},
            {"phase": "Refined Prototype 2 (Integrated UPI sandbox)", "deliverable": "Medium-Fidelity Bot with UPI sandbox integration", "security_gate": "NPCI UPI sandbox certification for test flows"},
            {"phase": "Specification Freeze & Production Build", "deliverable": "Validated Dialogue Specification handed to Agile-DevSecOps build team", "security_gate": "Product Head & Compliance Officer sign-off on final flows"}
        ]
    },

    # 12. Core Banking System Modernization (Iterative Model)
    {
        "domain": "Core Banking & Legacy Modernization",
        "stakeholder_queries": [
            "We are migrating our 20-year-old mainframe core banking system to a modern cloud-native platform. The migration must be done in phases without disrupting 10 million active accounts, with each phase independently deployable and rollback-capable.",
            "Our COBOL-based core banking needs to be replaced with a microservices architecture. We cannot do a big-bang cutover — we must deliver it in self-contained functional increments that each pass RBI audit before the next phase begins.",
            "Modernize our legacy core banking across 6 iterative phases: Accounts, Transactions, Lending, Deposits, Reporting, and Reconciliation. Each iteration delivers a fully tested, RBI-audited functional slice."
        ],
        "req_id_prefix": "REQ-IN-CBS",
        "statement": "The core banking modernization program shall replace the legacy mainframe system through six independently deployable functional iterations, each audited by CERT-In before the next iteration commences, ensuring zero disruption to existing account operations throughout the migration.",
        "categories": ["Core Banking Modernization", "Cloud Migration", "Legacy System Replacement"],
        "priority": "Must Have",
        "priority_reason": "RBI IT Framework for Banks mandates technology risk management and business continuity during core system changes.",
        "regulatory_citations": [
            "RBI IT Framework for Banks (Annex I — IT Governance) 2011",
            "RBI Guidelines on Business Continuity Planning (BCP) for Banks"
        ],
        "acceptance_criteria": "Each iteration delivers zero unplanned downtime; account balance reconciliation between legacy and new system shows zero discrepancy at end of each phase.",
        "dependencies": ["Legacy COBOL Data Migration ETL", "Cloud-Native Core Banking Platform", "Dual-run Reconciliation Engine"],
        "confidence_score": 0.95,
        "sdlc_model": "Iterative Model",
        "sdlc_confidence": 93,
        "sdlc_justification": "Core banking modernization involves replacing a complex, deeply integrated legacy system in stages — each iteration delivers a tested, audited functional slice (Accounts → Transactions → Lending) while keeping the legacy system running in parallel. The Iterative Model's cycle of plan-build-evaluate-refine is the only approach that manages migration risk while maintaining regulatory audit compliance at each stage.",
        "selection_rationale": {
            "why_this_model": "Iterative Model is chosen because core banking modernization is too large and complex for a single delivery, yet too risky for pure Agile sprints. Each iteration (e.g., Current Accounts module) is a self-contained, fully testable, RBI-auditable functional slice. After each iteration, lessons learned are incorporated into the next — the COBOL-to-microservices translation improves with each cycle, reducing risk progressively.",
            "why_not_alternative": "Agile-DevSecOps Hybrid is rejected for core banking migration because sprint-based continuous delivery conflicts with RBI IT Framework requirements for formal audit sign-off between migration phases — you cannot continuously deploy to production core banking systems without formal inter-phase audit gates.",
            "key_drivers": ["Zero-downtime phased migration", "RBI IT Framework inter-phase audit requirement", "Legacy parallel run reconciliation", "Progressive risk reduction per iteration", "10 million account continuity guarantee"]
        },
        "workflow": [
            {"phase": "Iteration 1: Current Accounts Module", "deliverable": "Cloud-Native Accounts Microservice + Legacy Dual-Run", "security_gate": "CERT-In Audit: Zero balance discrepancy across 1M accounts"},
            {"phase": "Iteration 2: Transaction Processing Engine", "deliverable": "Real-Time Transaction Processing with IMPS/NEFT/RTGS", "security_gate": "RBI IT Steering Committee Phase-2 Approval"},
            {"phase": "Iteration 3-5: Lending, Deposits, Reporting", "deliverable": "Module-by-module microservice with reconciliation reports", "security_gate": "Internal Audit sign-off before next phase kick-off"},
            {"phase": "Iteration 6: Legacy Decommission", "deliverable": "Mainframe Shutdown Certificate & Final Migration Report", "security_gate": "Board & RBI IT Examiners Final Certification"}
        ]
    },

    # 13. AI-Based Fraud Detection System (Spiral Model)
    {
        "domain": "Fraud Detection & Financial Crime AI",
        "stakeholder_queries": [
            "We are building an AI/ML-based real-time fraud detection system for UPI and card transactions. The models are experimental, regulatory acceptance of AI decisions in banking is unclear, and the risk of false positives blocking legitimate transactions is very high.",
            "Develop a deep learning fraud detection engine that flags suspicious UPI transactions in under 200ms. The technical risk is high — model accuracy, regulatory compliance under RBI AI guidelines, and false-positive impact on customers are all uncertain.",
            "Build an ML fraud scoring system for our NBFC lending platform. We need to manage significant uncertainty around model performance, RBI stance on AI-based credit/fraud decisions, and data quality risks before full deployment."
        ],
        "req_id_prefix": "REQ-IN-FRAUD",
        "statement": "The AI-based fraud detection engine shall score UPI and card transactions in real-time under 200ms, using ensemble ML models with explainable AI outputs, with each development spiral conducting formal risk analysis of model accuracy, regulatory compliance, and customer impact before proceeding.",
        "categories": ["AI/ML Fraud Detection", "Real-Time Risk Scoring", "Regulatory AI Governance"],
        "priority": "Must Have",
        "priority_reason": "RBI Guidelines on Model Risk Management mandate formal validation and risk assessment for AI/ML models used in financial decision-making.",
        "regulatory_citations": [
            "RBI Draft Guidelines on Model Risk Management for AI/ML in Financial Services",
            "RBI/2021-22/68 Cyber Security Controls for Third Party ATM Switch Application Service Providers"
        ],
        "acceptance_criteria": "Model false-positive rate < 0.1%; all AI decisions have explainable output logged per RBI AI governance requirements; each spiral passes independent model validation.",
        "dependencies": ["Feature Engineering Pipeline", "ML Model Training Infrastructure (GPU)", "Explainable AI (XAI) Layer", "Real-Time Scoring API"],
        "confidence_score": 0.92,
        "sdlc_model": "Spiral Model",
        "sdlc_confidence": 91,
        "sdlc_justification": "AI-based fraud detection carries multiple high-risk dimensions simultaneously: model accuracy risk (false positives blocking real transactions), regulatory risk (RBI AI governance requirements are evolving), and data quality risk (historical fraud data imbalance). The Spiral Model's mandatory risk analysis at each loop ensures no risk is ignored before the next development phase begins.",
        "selection_rationale": {
            "why_this_model": "Spiral Model is chosen because AI fraud detection involves multiple compounding risks that must each be formally evaluated before committing to the next development phase. Each spiral loop explicitly: (1) determines objectives for this phase, (2) identifies and resolves risks (model accuracy, regulatory compliance, data quality), (3) develops and verifies the phase output, and (4) plans the next spiral. This is the only methodology that makes risk evaluation a first-class deliverable — not an afterthought.",
            "why_not_alternative": "Agile-DevSecOps Hybrid is rejected because Agile assumes sprint-level risk tolerance — but deploying an inaccurate fraud model to production even for one sprint could incorrectly block thousands of legitimate UPI transactions, causing regulatory penalties under RBI Consumer Protection guidelines. Risk must be formally assessed and mitigated before any production exposure.",
            "key_drivers": ["High technical risk: AI model accuracy uncertainty", "Regulatory risk: Evolving RBI AI governance", "Customer impact risk: False positives blocking transactions", "Data quality risk: Imbalanced fraud training data", "Mandatory formal risk assessment before each spiral"]
        },
        "workflow": [
            {"phase": "Spiral 1: Risk Analysis & POC", "deliverable": "Risk Register + ML Baseline Model on Historical Data", "security_gate": "Model Validation Committee: Baseline accuracy > 95% on holdout set"},
            {"phase": "Spiral 2: Production-Sim Risk Mitigation", "deliverable": "Ensemble Model with XAI + Synthetic Traffic Stress Test", "security_gate": "False-positive rate < 0.1% verified on 10M synthetic transactions"},
            {"phase": "Spiral 3: Shadow Mode Deployment", "deliverable": "Real-time Scoring running parallel (not blocking) for 30 days", "security_gate": "RBI Model Risk Validation: Independent third-party model audit"},
            {"phase": "Spiral 4: Production Cut-over", "deliverable": "Live Fraud Scoring with Human-in-Loop Override Panel", "security_gate": "CISO, Chief Risk Officer & RBI IT Examiner Final Sign-off"}
        ]
    },

    # 14. Internal Regulatory Reporting Dashboard (RAD Model)
    {
        "domain": "Internal Regulatory Reporting & Analytics",
        "stakeholder_queries": [
            "We need an internal dashboard for our compliance team to generate RBI OSMOS regulatory reports, track SLAs for STR filings, and monitor Basel III capital ratios. This is an internal tool with no customer data — we need it within 6 weeks.",
            "Build a compliance officer dashboard that auto-generates CRILC data submissions, monitors CRR/SLR requirements, and flags upcoming regulatory filing deadlines. Stakeholders are available for daily feedback — we need rapid delivery.",
            "Our internal audit team needs a reporting tool to generate SEBI SCORES escalation reports and monitor compliance across 12 regulatory domains. Users will define exact requirements as we build — fast delivery is critical."
        ],
        "req_id_prefix": "REQ-IN-REGDASH",
        "statement": "The internal regulatory reporting dashboard shall enable the compliance team to generate RBI OSMOS, CRILC, and SEBI SCORES reports with automated deadline tracking, accessible only to authorized compliance officers via role-based access control.",
        "categories": ["Internal Compliance Tooling", "Regulatory Reporting Automation", "Analytics"],
        "priority": "Should Have",
        "priority_reason": "Internal operational efficiency tool; regulatory reports themselves are already governed — the dashboard accelerates their preparation.",
        "regulatory_citations": [
            "RBI OSMOS (Online Returns Filing System) Reporting Guidelines",
            "SEBI SCORES Platform Guidelines for Registered Intermediaries"
        ],
        "acceptance_criteria": "All 12 regulatory report templates auto-populated from existing data warehouse within 2 clicks; deadline alerts triggered 5 business days in advance.",
        "dependencies": ["Data Warehouse / Data Lake", "Role-Based Access Control Module", "Email Alerting System"],
        "confidence_score": 0.88,
        "sdlc_model": "RAD (Rapid Application Development)",
        "sdlc_confidence": 89,
        "sdlc_justification": "Internal compliance dashboards have known users (compliance officers), available stakeholders for daily feedback, reusable component libraries (chart widgets, report templates), and tight delivery timelines. RAD's component-based construction with continuous stakeholder workshops delivers a working tool rapidly without the overhead of formal specification phases needed for customer-facing regulatory systems.",
        "selection_rationale": {
            "why_this_model": "RAD (Rapid Application Development) is chosen because internal regulatory dashboards have all the conditions RAD requires: (1) clearly identified users who are available for daily feedback workshops, (2) no customer PII risk requiring formal security gates, (3) reusable report template components that can be assembled quickly, (4) a 6-week delivery mandate from the compliance team. RAD's JRP (Joint Requirements Planning) → JD (Joint Application Design) → construction → cutover cycle achieves this.",
            "why_not_alternative": "V-Model is rejected because internal compliance dashboards do not need the formal verification artifacts, audit sign-offs, and sequential phase gates that V-Model mandates — those are reserved for customer-facing, high-risk regulatory systems. Applying V-Model here would add 3-4 months of overhead to a 6-week tool delivery.",
            "key_drivers": ["6-week internal delivery mandate", "Daily stakeholder feedback available", "Reusable report template components", "Internal tool: no customer PII exposure", "Compliance team available for Joint Application Design workshops"]
        },
        "workflow": [
            {"phase": "JRP Workshop (Joint Requirements Planning)", "deliverable": "Prioritized Report Catalog & Dashboard Wireframes (2 days)", "security_gate": "Chief Compliance Officer sign-off on scope"},
            {"phase": "JAD Workshop & Component Assembly", "deliverable": "Working dashboard with 6 core report templates", "security_gate": "Stakeholder demo: All 6 templates verified by compliance team"},
            {"phase": "Rapid Construction & Integration", "deliverable": "Fully integrated dashboard with data warehouse connectors", "security_gate": "IT Security: RBAC configuration verified, no unauthorized data access"},
            {"phase": "User Acceptance & Cutover", "deliverable": "Production dashboard with user training", "security_gate": "Head of Compliance formal acceptance sign-off"}
        ]
    },

    # 15. RBI CBDC (Digital Rupee) Integration (Incremental Model)
    {
        "domain": "Central Bank Digital Currency (CBDC)",
        "stakeholder_queries": [
            "RBI has mandated our bank to integrate with the Digital Rupee (e-Rupee) retail CBDC pilot. The integration has 4 distinct phases: wallet setup, P2P transfer, merchant payments, and programmable CBDC — each phase requires separate RBI pilot approval before next phase begins.",
            "Integrate the RBI Digital Rupee CBDC with our mobile banking app in increments. Phase 1 is e-Rupee wallet; Phase 2 adds P2P transfers; Phase 3 enables merchant payments via QR; Phase 4 adds programmable CBDC for government disbursements.",
            "Build the bank's interface with RBI's Digital Rupee infrastructure. Each increment (wallet, transfers, QR payments, programmable CBDC) must be independently certified by RBI before the next increment is built."
        ],
        "req_id_prefix": "REQ-IN-CBDC",
        "statement": "The Digital Rupee (e-Rupee) integration shall be delivered in four independent increments — e-Rupee Wallet, P2P Transfer, Merchant QR Payment, and Programmable CBDC — each certified by RBI's CBDC pilot authority before the subsequent increment commences development.",
        "categories": ["Central Bank Digital Currency (CBDC)", "Digital Rupee Integration", "RBI Pilot Program"],
        "priority": "Must Have",
        "priority_reason": "RBI Concept Note on Central Bank Digital Currency (October 2022) mandates phased, controlled pilot integration for participant banks.",
        "regulatory_citations": [
            "RBI Concept Note on Central Bank Digital Currency (October 2022)",
            "RBI CBDC Retail Pilot Program Guidelines for Pilot Banks"
        ],
        "acceptance_criteria": "Each increment independently deployable and functional; RBI CBDC pilot authority certification obtained before next increment build begins.",
        "dependencies": ["RBI CBDC Core Infrastructure API", "e-Rupee Token Ledger", "Programmable CBDC Smart Contract Engine"],
        "confidence_score": 0.94,
        "sdlc_model": "Incremental Model",
        "sdlc_confidence": 94,
        "sdlc_justification": "CBDC integration with RBI's Digital Rupee infrastructure follows a mandated phased pilot structure where each increment (Wallet → P2P → Merchant → Programmable) requires independent RBI certification before the next begins. The Incremental Model maps exactly to this structure — each increment is a fully functional, tested, and certified vertical slice of the final system.",
        "selection_rationale": {
            "why_this_model": "Incremental Model is chosen because RBI's CBDC pilot program is itself structured as independent certification phases — the bank cannot begin P2P transfer development until the e-Rupee Wallet increment is certified by RBI. Each increment delivers a fully functional, independently deployable and auditable capability. This is the defining characteristic of Incremental delivery: partial but complete functional builds, not partial functionality.",
            "why_not_alternative": "Iterative Model is rejected because Iterative revisits and refines the same functionality across cycles — but CBDC increments are distinct new capabilities (Wallet ≠ P2P ≠ Merchant QR), not refinements of the same feature. The Incremental Model's 'new capability per release' structure maps precisely to RBI's phased pilot certification schedule.",
            "key_drivers": ["RBI CBDC pilot phase certification mandate", "Each increment independently deployable", "Distinct capabilities per increment (not refinements)", "New financial infrastructure with unknown technical risks per phase", "Controlled pilot program — cannot fast-track phases"]
        },
        "workflow": [
            {"phase": "Increment 1: e-Rupee Wallet", "deliverable": "Digital Rupee wallet with balance, top-up, and redemption", "security_gate": "RBI CBDC Pilot Authority: Increment 1 Certification"},
            {"phase": "Increment 2: P2P Transfer", "deliverable": "Person-to-Person Digital Rupee transfer with offline capability", "security_gate": "RBI CBDC Pilot Authority: Increment 2 Certification"},
            {"phase": "Increment 3: Merchant QR Payment", "deliverable": "CBDC QR payment at merchant PoS with UPI interoperability", "security_gate": "NPCI & RBI Joint Interoperability Certification"},
            {"phase": "Increment 4: Programmable CBDC", "deliverable": "Smart Contract engine for government welfare CBDC disbursements", "security_gate": "Ministry of Finance & RBI Joint Certification"}
        ]
    },

    # 16. New Digital Bank Product (Dual-Track Agile)
    {
        "domain": "New Digital Bank Product Development",
        "stakeholder_queries": [
            "We are launching a completely new zero-balance savings account product for Gen-Z customers with instant KYC, zero fees, UPI autopay, and investment integration. We need to simultaneously discover what features customers actually want AND build the technical platform — two parallel tracks.",
            "Our bank wants to build a new micro-savings product for gig economy workers. Product discovery (user research, feature prioritization, regulatory feasibility) must run in parallel with platform engineering — we cannot wait for discovery to complete before building.",
            "Launch a new digital-only SME banking product. Run a discovery track (customer interviews, regulatory analysis, RBI Small Finance Bank guidelines review) simultaneously with a delivery track (API platform, KYC microservice, accounting engine) to compress time-to-market."
        ],
        "req_id_prefix": "REQ-IN-NEOBANK",
        "statement": "The new digital banking product platform shall be developed using parallel discovery and delivery tracks — the discovery track continuously researches customer needs and regulatory feasibility while the delivery track builds the core technical platform, with weekly synchronization to ensure discovered requirements are immediately incorporated into delivery.",
        "categories": ["New Product Development", "Digital Banking", "Lean Product Discovery"],
        "priority": "Should Have",
        "priority_reason": "RBI guidelines on Small Finance Banks and Payments Banks mandate product documentation and regulatory approval before customer launch.",
        "regulatory_citations": [
            "RBI Guidelines for Licensing of Small Finance Banks in the Private Sector",
            "RBI Master Direction on KYC (for Digital Onboarding of New Customers)"
        ],
        "acceptance_criteria": "Discovery track produces validated customer personas and regulatory feasibility report within 4 sprints; delivery track produces working platform within 8 sprints; both tracks synchronized at weekly integration checkpoints.",
        "dependencies": ["User Research Panel", "Regulatory Sandbox (RBI Regulatory Sandbox Program)", "Core Banking API Platform", "Digital KYC Engine"],
        "confidence_score": 0.89,
        "sdlc_model": "Dual-Track Agile",
        "sdlc_confidence": 88,
        "sdlc_justification": "Building a brand-new digital banking product requires simultaneous discovery (what should we build? what do customers want? what does RBI allow?) and delivery (building the technical platform). Dual-Track Agile runs product discovery and platform engineering in parallel sprints, with weekly synchronization — cutting time-to-market while ensuring the platform reflects real customer and regulatory needs.",
        "selection_rationale": {
            "why_this_model": "Dual-Track Agile is chosen because launching a new digital banking product involves two fundamentally different work streams that must run in parallel: (1) Discovery Track — customer interviews, regulatory feasibility under RBI guidelines, competitive analysis, and feature prioritization; (2) Delivery Track — building the KYC engine, API platform, and accounting microservices. Running them sequentially (discover first, then build) wastes 4-6 months. Dual-Track compresses this by running both simultaneously with weekly synchronization.",
            "why_not_alternative": "Standard Agile-DevSecOps Hybrid is rejected because it assumes a single team working from a ready backlog — but for a genuinely new banking product, the backlog itself is unknown until discovery research is completed. A single-track Agile approach would force premature feature commitment before customer validation, leading to costly rework.",
            "key_drivers": ["Simultaneous product discovery and platform build", "Compressed time-to-market for new product launch", "RBI regulatory feasibility research running in parallel", "Weekly discovery-delivery synchronization sprint rhythm", "Unknown customer needs requiring parallel research"]
        },
        "workflow": [
            {"phase": "Dual Sprint 1-2: Parallel Kickoff", "deliverable": "Discovery: Customer personas + RBI sandbox application | Delivery: Core API scaffolding", "security_gate": "Product-Compliance alignment: RBI sandbox approval obtained"},
            {"phase": "Dual Sprint 3-6: Core Discovery & Build", "deliverable": "Discovery: Validated features + regulatory constraints | Delivery: KYC + Accounts microservices", "security_gate": "Weekly sync gate: No delivery sprint builds unvalidated features"},
            {"phase": "Dual Sprint 7-8: Integration & Beta", "deliverable": "Beta product with 500 invited customers in RBI sandbox", "security_gate": "RBI Sandbox Program: No regulatory violations in beta"},
            {"phase": "Production Launch", "deliverable": "Regulatory-approved product launch with full KYC and RBI product filing", "security_gate": "RBI Product Filing Accepted & CISO Security Sign-off"}
        ]
    },

    # 17. One-Time RTGS Message Format Migration (Waterfall Model)
    {
        "domain": "Payment Infrastructure & RTGS Modernization",
        "stakeholder_queries": [
            "RBI has mandated that all RTGS participant banks must migrate from the legacy ISO 8583 message format to ISO 20022 XML by the fixed regulatory deadline. The specification is 100% defined by SWIFT/RBI — there is zero ambiguity, zero requirement discovery, and zero iteration needed. We just need to build exactly to spec and deliver by the deadline.",
            "Our bank must complete the one-time migration of RTGS/NEFT message formats to ISO 20022. The RBI circular has published the exact field mappings, XML schemas, and validation rules. This is a build-to-specification project with a hard regulatory cutover date — no room for iterative discovery.",
            "Implement the RBI-mandated ISO 20022 migration for our payment systems. The SWIFT MX message schemas are published, the field-by-field mapping from MT to MX is defined, and the IDRBT testing schedule is fixed. This is a classical specification-driven, single-delivery project."
        ],
        "req_id_prefix": "REQ-IN-RTGS",
        "statement": "The RTGS/NEFT payment gateway shall migrate all outbound and inbound message processing from ISO 8583 (legacy) to ISO 20022 (MX) XML format by the RBI-mandated cutover date, with zero message loss and full backward compatibility during the parallel run period.",
        "categories": ["Payment Infrastructure Modernization", "ISO 20022 Migration", "RTGS/NEFT Compliance"],
        "priority": "Must Have",
        "priority_reason": "RBI circular mandates ISO 20022 adoption for all RTGS participants with a non-negotiable cutover deadline.",
        "regulatory_citations": [
            "RBI/2019-20/194 DPSS.CO.RTGS No.1862/04.04.016/2019-20 (ISO 20022 Migration)",
            "SWIFT Standards MT to MX Migration Guidelines for Indian Banks"
        ],
        "acceptance_criteria": "Zero message format validation errors against RBI-published ISO 20022 XSD; zero transaction loss during 30-day parallel run; successful IDRBT mock cutover drill.",
        "dependencies": ["SWIFT Alliance Gateway Upgrade", "ISO 20022 Message Parser Library", "IDRBT Test Environment Access"],
        "confidence_score": 0.97,
        "sdlc_model": "Waterfall",
        "sdlc_confidence": 96,
        "sdlc_justification": "ISO 20022 migration is a textbook Waterfall project — the specification is 100% defined by SWIFT and RBI with zero ambiguity, the field mappings are published and frozen, the XSD schemas are immutable, and the cutover date is a hard regulatory deadline. There is literally nothing to discover, iterate, or prototype — the only task is to build exactly to the published specification and deliver by the deadline.",
        "selection_rationale": {
            "why_this_model": "Waterfall is chosen because this is the rare, genuine case where ALL Waterfall preconditions are met: (1) requirements are 100% complete and frozen — the ISO 20022 XSD schemas are published by SWIFT and mandated by RBI, (2) there is zero requirement uncertainty or user discovery needed, (3) the technology stack is proven (XML parsing, SWIFT Alliance), (4) the deliverable is a single, well-defined cutover event. Waterfall's sequential Requirements → Design → Implementation → Testing → Deployment flow maps precisely to this project's structure.",
            "why_not_alternative": "Agile-DevSecOps Hybrid is rejected because there are no evolving requirements to iterate on — the ISO 20022 schema is frozen by SWIFT international standards body. Running Agile sprints on a fixed specification adds process overhead (sprint planning, retrospectives, backlog grooming) without any corresponding benefit, since the scope will never change mid-project.",
            "key_drivers": ["100% frozen specification (SWIFT ISO 20022 XSD)", "Zero requirement uncertainty", "Hard regulatory cutover deadline", "Single delivery event (not incremental)", "Proven technology with no technical risk"]
        },
        "workflow": [
            {"phase": "Requirements Analysis (ISO 20022 Schema Mapping)", "deliverable": "Complete MT-to-MX Field Mapping Document & Gap Analysis", "security_gate": "Payment Systems Head & Compliance Officer Sign-off"},
            {"phase": "System Design (Message Transformation Architecture)", "deliverable": "MX Message Parser & Transformer Design Document", "security_gate": "Enterprise Architecture Review Board Approval"},
            {"phase": "Implementation & Unit Testing", "deliverable": "ISO 20022 Message Processing Engine with 100% XSD Compliance", "security_gate": "Zero XSD validation errors across all 47 MT message types"},
            {"phase": "IDRBT Mock Cutover & Production Deployment", "deliverable": "Successful 30-Day Parallel Run Report & RBI Cutover Certification", "security_gate": "IDRBT Conformance Certificate & RBI Cutover Approval"}
        ]
    },

    # 18. IRDAI Statutory Returns Filing System (Waterfall Model)
    {
        "domain": "Insurance Regulatory Compliance & Statutory Filing",
        "stakeholder_queries": [
            "Build a system to generate and file IRDAI statutory returns (Form A, Form B, Form H, Form I) exactly as per the IRDAI (Preparation of Financial Statements and Auditor's Report of Insurance Companies) Regulations. The forms, fields, and filing format are 100% defined by IRDAI regulation — zero discovery needed.",
            "We need a one-time system build for IRDAI quarterly and annual regulatory returns filing. The XML schemas, field definitions, and calculation formulae are all published in the IRDAI Master Circular. This is a pure build-to-spec project with no iterative discovery.",
            "Implement the IRDAI statutory returns filing system for our insurance company. Every field, validation rule, and submission format is defined by IRDAI regulation. We need to build exactly to specification and deploy before the next quarterly filing deadline."
        ],
        "req_id_prefix": "REQ-IN-IRDAI",
        "statement": "The IRDAI statutory returns filing system shall generate Form A (Revenue Account), Form B (Profit & Loss), Form H (Analytical Ratios), and Form I (Investment Schedule) in the exact format prescribed by IRDAI regulations, with automated validation against IRDAI-published calculation rules before submission.",
        "categories": ["Insurance Regulatory Filing", "IRDAI Compliance", "Statutory Reporting"],
        "priority": "Must Have",
        "priority_reason": "IRDAI (Preparation of Financial Statements) Regulations mandate statutory returns filing with prescribed formats — non-compliance results in regulatory penalties and license risk.",
        "regulatory_citations": [
            "IRDAI (Preparation of Financial Statements and Auditor's Report of Insurance Companies) Regulations, 2002",
            "IRDAI Master Circular on Statutory Returns Filing (Annual & Quarterly)"
        ],
        "acceptance_criteria": "All statutory forms match IRDAI-prescribed format with zero field omissions; automated validation catches 100% of calculation errors before submission; successful test filing accepted by IRDAI portal.",
        "dependencies": ["Actuarial Data Warehouse", "IRDAI Filing Portal API", "Chartered Accountant Validation Module"],
        "confidence_score": 0.96,
        "sdlc_model": "Waterfall",
        "sdlc_confidence": 95,
        "sdlc_justification": "IRDAI statutory returns have completely fixed, regulation-defined formats with zero ambiguity in requirements. Every field, every calculation formula, every validation rule is published in IRDAI regulations. This is a pure build-to-specification project where Waterfall's sequential approach (understand regulation → design system → implement forms → validate → deploy) is optimal because there is nothing to discover or iterate on.",
        "selection_rationale": {
            "why_this_model": "Waterfall is chosen because IRDAI statutory returns filing is the purest form of build-to-specification software — every form field, every calculation formula (solvency ratio, incurred claims ratio, investment yield), and every XML tag is prescribed by IRDAI regulation. Requirements cannot change mid-project because they are statutory law. The sequential Waterfall flow (Regulation Analysis → Form Design → Implementation → CA Validation → IRDAI Test Filing → Production) is the most efficient path because no phase produces ambiguity that would require iteration.",
            "why_not_alternative": "Agile is rejected because there are literally zero stories to discover — the 'product owner' is IRDAI regulation itself, and it does not attend sprint reviews or change requirements between sprints. Spiral Model is rejected because there is no technical or requirements risk to analyze — the forms, validations, and filing APIs are well-established and used by every insurance company in India.",
            "key_drivers": ["100% regulation-defined form specifications", "Zero requirements uncertainty", "Fixed quarterly filing deadline", "Statutory calculation formulae (no business logic ambiguity)", "One-time system build with annual maintenance"]
        },
        "workflow": [
            {"phase": "Regulatory Requirements Extraction", "deliverable": "Complete IRDAI Form Field Catalogue with Calculation Formulae", "security_gate": "Chief Actuary & Compliance Head Sign-off on Field Completeness"},
            {"phase": "System Design & Data Mapping", "deliverable": "Actuarial Data Warehouse → IRDAI Form Field Mapping Document", "security_gate": "IT Architecture Review: All data sources identified and mapped"},
            {"phase": "Implementation & Unit Testing", "deliverable": "Form Generation Engine with Automated Calculation Validation", "security_gate": "CA (Chartered Accountant) Parallel Calculation Verification: Zero Discrepancies"},
            {"phase": "IRDAI Test Filing & Production Deployment", "deliverable": "Successful Test Filing Accepted by IRDAI Portal", "security_gate": "Appointed Actuary & Statutory Auditor Formal Sign-off"}
        ]
    }
]



def synthesize_indian_datasets(target_count=350):
    print(f"[INFO] Synthesizing {target_count} complex, domain-specific Indian Banking & Finance training samples...")
    extraction_samples = []
    sdlc_samples = []

    ext_prompts = [
        "Transform the following Indian banking stakeholder dialogue into a strict JSON requirement schema with RBI/NPCI/UIDAI regulatory citations and MoSCoW prioritization:",
        "Extract the structured software requirement from this financial engineering interview turn, classifying statutory priority and Indian regulatory compliance frameworks:",
        "Analyze this banking requirement dialogue, outputting a formal JSON schema with acceptance criteria and applicable RBI/SEBI/DPDP statutory citations:",
        "Convert this Indian FinTech stakeholder request into a verifiable, audit-ready engineering requirement specification schema:",
        "Structure this digital banking requirement according to Indian regulatory compliance standards (RBI Master Directions, NPCI, UIDAI ADV):"
    ]

    sdlc_prompts = [
        "Based on the following Indian banking requirement and its statutory compliance profile, recommend an optimal SDLC methodology and phased stage-gate roadmap:",
        "Evaluate regulatory risk, compliance criticality, and architectural dependencies to generate a tailored SDLC execution plan for this Indian financial system requirement:",
        "Propose and justify a structured SDLC framework for the given RBI/NPCI-regulated banking requirement, including confidence score and security checkpoints:",
        "Recommend an optimal Software Development Life Cycle model and stage-gate workflow complying with RBI Cyber Security and DevSecOps guidelines:"
    ]

    # --- Stratified Sampling: equal representation per SDLC model ---
    from collections import defaultdict
    model_groups = defaultdict(list)
    for s in INDIAN_BANKING_SCENARIOS:
        model_groups[s["sdlc_model"]].append(s)

    sdlc_model_names = sorted(model_groups.keys())
    num_models = len(sdlc_model_names)
    samples_per_model = target_count // num_models
    remainder = target_count % num_models

    # Build a balanced scenario list: each model gets ~equal samples
    balanced_scenarios = []
    for idx, model_name in enumerate(sdlc_model_names):
        count = samples_per_model + (1 if idx < remainder else 0)
        scenarios_for_model = model_groups[model_name]
        for j in range(count):
            balanced_scenarios.append(scenarios_for_model[j % len(scenarios_for_model)])

    random.shuffle(balanced_scenarios)

    print(f"[INFO] Stratified sampling: {num_models} SDLC models, ~{samples_per_model} samples each")
    for mn in sdlc_model_names:
        c = sum(1 for s in balanced_scenarios if s["sdlc_model"] == mn)
        print(f"  → {mn}: {c} samples")

    for i in range(target_count):
        scenario = balanced_scenarios[i]
        query = random.choice(scenario["stakeholder_queries"])
        req_id = f"{scenario['req_id_prefix']}-{1000 + i}"

        # Natural variations in stakeholder dialogue
        variations = [
            f" [Audit Finding #{1000 + i}]: This was flagged during our statutory RBI risk assessment.",
            f" [Circular Ref #{2024 + i%5}]: Must be deployed in Phase {i%3 + 1} with zero downtime.",
            " Note: All logs must be stored on domestic Indian servers with immutable hash chains.",
            " High priority item for upcoming concurrent audit and NPCI compliance review.",
            ""
        ]
        input_dialogue = query + random.choice(variations)

        # Extraction Output
        ext_json = {
            "req_id": req_id,
            "statement": scenario["statement"],
            "categories": scenario["categories"],
            "priority": scenario["priority"],
            "priority_justification": scenario["priority_reason"],
            "regulatory_citations": scenario["regulatory_citations"],
            "acceptance_criteria": scenario["acceptance_criteria"],
            "dependencies": scenario["dependencies"],
            "confidence_score": scenario["confidence_score"]
        }

        ext_prompt = random.choice(ext_prompts)
        extraction_samples.append({
            "instruction": "You are the Extraction & Classification Agent for Indian banking and financial software systems. Convert the stakeholder dialogue into a strict JSON requirement schema with MoSCoW prioritization and RBI/NPCI/UIDAI/DPDP statutory citations. Output ONLY valid JSON.",
            "input": f"{ext_prompt}\n\nStakeholder Dialogue:\n\"{input_dialogue}\"",
            "output": json.dumps(ext_json, indent=2)
        })

        # SDLC Output — simplified: model + reasoning + alternatives
        rationale = scenario["selection_rationale"]

        # Build alternatives from why_not_* keys in selection_rationale
        key_model_map = {
            "why_not_agile": "Agile-DevSecOps Hybrid",
            "why_not_agile_devsecops": "Agile-DevSecOps Hybrid",
            "why_not_waterfall": "Waterfall",
            "why_not_v_model": "V-Model",
        }
        alternatives = []
        for key, val in rationale.items():
            if not key.startswith("why_not"):
                continue
            if key in key_model_map:
                alternatives.append({"model": key_model_map[key], "why_not": val})
            elif key == "why_not_alternative":
                # Parse model name from "ModelName is rejected because..."
                import re
                match = re.match(r'^([\w\s\-\(\)/]+?)\s+is rejected', val)
                alt_model = match.group(1).strip() if match else "Common Alternative"
                alternatives.append({"model": alt_model, "why_not": val})

        sdlc_json = {
            "recommended_model": scenario["sdlc_model"],
            "reasoning": rationale["why_this_model"],
            "alternatives_considered": alternatives
        }

        sdlc_prompt = random.choice(sdlc_prompts)
        sdlc_input_text = (
            f"{sdlc_prompt}\n\n"
            f"Requirement ID: {req_id}\n"
            f"Requirement Statement:\n\"{scenario['statement']}\"\n"
            f"Categories: {', '.join(scenario['categories'])}\n"
            f"Priority: {scenario['priority']}\n"
            f"Regulatory Framework: {', '.join(scenario['regulatory_citations'])}"
        )

        sdlc_samples.append({
            "instruction": "You are the SDLC Recommendation Agent for Indian financial institutions. Analyze the requirement and output a JSON with: recommended_model (SDLC model name), reasoning (why this model fits), and alternatives_considered (list of rejected models with why_not explanation). Output ONLY valid JSON.",
            "input": sdlc_input_text,
            "output": json.dumps(sdlc_json, indent=2)
        })

    # Save to disk
    ext_file = "data/training/agent_extraction.jsonl"
    with open(ext_file, "w", encoding="utf-8") as f:
        for item in extraction_samples:
            f.write(json.dumps(item) + "\n")

    sdlc_file = "data/training/agent_sdlc.jsonl"
    with open(sdlc_file, "w", encoding="utf-8") as f:
        for item in sdlc_samples:
            f.write(json.dumps(item) + "\n")

    print(f"[SUCCESS] Synthesized {len(extraction_samples)} Indian Banking samples in {ext_file}")
    print(f"[SUCCESS] Synthesized {len(sdlc_samples)} Indian Banking samples in {sdlc_file}")


if __name__ == "__main__":
    synthesize_indian_datasets(350)
