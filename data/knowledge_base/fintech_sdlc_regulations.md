# SDLC Model Selection for Indian Financial Systems

## 1. Key Principle

Indian financial regulators (RBI, SEBI, NPCI, CERT-In, IRDAI) do not prescribe a named SDLC model such as Waterfall, V-Model or Agile. They prescribe controls and evidence: secure development practices, change management, pre-production testing, independent security assessment, segregation of duties, audit trails and incident reporting.

So the right SDLC model depends on the system's risk profile and how it changes over time, and on whether the model can produce the evidence regulators and auditors expect. Any model, including Agile, is acceptable if it produces that evidence. The correct statement is "this system's control burden is best met by a gated, V-Model-style process", not "the regulator mandates the V-Model".

## 2. Regulatory Control Sources

| Regulator | Instrument | Relevance to SDLC |
|---|---|---|
| RBI | Cyber Security Framework in Banks (2016) | Secure configuration, VAPT, change control, SOC, incident response |
| RBI | Master Direction on IT Governance, Risk, Controls and Assurance Practices (2023) | IT governance, SDLC and change management, IT audit, BCP/DR |
| RBI | Master Direction on Outsourcing of IT Services (2023) | Governance of vendor-built and third-party software |
| RBI | Master Direction on Digital Payment Security Controls (2021) | Secure development and testing of banking and payment apps |
| RBI | KYC Master Direction (V-CIP provisions) | Video-KYC process controls and audit-ready recording |
| RBI | Digital Lending Directions (2025) | Lending app and LSP governance, data handling |
| RBI | Regulatory Sandbox framework (2019) | Time-boxed, controlled live testing of innovations |
| SEBI | Cybersecurity and Cyber Resilience Framework, CSCRF (2024) | Governance, VAPT, audit and incident response for SEBI-regulated entities |
| SEBI | Algo trading framework for retail participation (2025) | Broker and exchange controls, algo registration and audit |
| NPCI | UPI and other operating circulars, security guidelines | Member and PSP app testing and certification before go-live |
| CERT-In | Directions of April 2022 | 6-hour incident reporting, 180-day log retention in India |
| CERT-In | Empanelled auditor list | Source of empanelled security auditors for VAPT |
| MeitY | Digital Personal Data Protection Act 2023 and Rules | Consent, data minimization, breach notification |

Baseline standards commonly referenced: ISO/IEC 27001, OWASP ASVS and Top 10, FIPS 140-3 (FIPS 140-2 certificates are being retired).

Verify exact circular numbers, dates and amendments against the regulator's own site before citing them in formal work. These instruments are updated often.

## 3. Factors That Drive the Choice

| Factor | Low end | High end |
|---|---|---|
| Financial finality | Reversible, informational | Irreversible money movement or ledger posting |
| Blast radius | One user | Systemic, market-wide, interbank |
| Change reversibility | Instant rollback, feature flags | Cannot be patched after deployment (hardware, firmware) |
| Requirement volatility | Stable statutory specification | Experimental or UX-driven |
| Formal attestation load | Internal sign-off | Regulator, exchange or NPCI certification per release |
| Real-time criticality | Batch tolerant | Millisecond latency, 24x7 |
| Sensitive data exposure | Public data | Credentials, PII, KYC, cryptographic keys |
| Hardware dependence | Pure software | Certified physical module |

## 4. Rules of Thumb

1. Hardware or firmware that cannot be changed after release: Waterfall / staged-gate.
2. High financial finality, systemic impact, per-release attestation, stable requirements: V-Model with DevSecOps gates.
3. High assurance but some ongoing change: V-Model core with Agile increments released through gates.
4. Experimental, low financial exposure, sandbox or pilot: Prototyping / iterative.
5. Customer-facing, frequently changing, reversible: Agile-DevSecOps with automated compliance gates.
6. Anything unclear: hybrid, and flag for human review.

If the system handles credentials, PII or KYC data, add security gates (SAST, SCA, secret scanning, VAPT) whatever the model. If it is vendor-built or outsourced, add vendor governance and audit rights.

## 5. Model Profiles

### 5.1 V-Model with DevSecOps Gates

**Typical systems:** ATM and payment switch core and host interface (ISO 8583), exchange matching engines and algorithmic trading gateways (FIX), RTGS and high-value payment systems (ISO 20022), core banking general ledger and interest engines, clearing and settlement systems.

**Why it fits:** high financial finality, systemic blast radius and per-release attestation. Bidirectional traceability from requirement to design to test to evidence is the most direct way to demonstrate the required testing.

**Evidence produced:** requirements traceability matrix, unit, integration, system and UAT records, performance and DR test results, independent VAPT by a CERT-In empanelled auditor where required, change approval records and a rollback plan.

**Note:** modern V-Model delivery runs inside CI/CD with automated tests at every stage. It means gated verification, not slow manual cycles.

### 5.2 Agile-DevSecOps Hybrid

**Typical systems:** mobile banking apps, UPI client and PSP app interfaces, digital lending apps and portals, V-CIP video-KYC workflows, agent banking and field-officer apps, onboarding journeys.

**Why it fits:** frequent product and UX change, easy rollback through feature flags and staged rollout, and sensitive data that calls for strong automated security controls.

**Minimum CI/CD gates:**
1. SAST that blocks builds on High or Critical findings.
2. Software composition analysis for open-source CVEs, with an SBOM.
3. Secret scanning on commits and pipelines.
4. DAST or mobile app security testing before release.
5. Maker-checker approval for production deployment (segregation of duties).
6. Immutable audit log of build, approval and deployment.

**Note:** for regulated apps, NPCI or RBI-required testing and certification may still apply per release. Agile does not remove those gates. It has to schedule them.

### 5.3 Waterfall / Staged-Gate

**Typical systems:** custom HSM firmware and modules (FIPS 140-3 validation, or legacy FIPS 140-2), physical vault and currency chest access controllers, fixed firmware sensor boards.

**Why it fits:** the specification and cryptographic interfaces must be frozen before fabrication or validation, and change after release is costly or impossible.

**Note:** surrounding software, such as management applications and integration layers, can use a different model. Choose per component, not per product.

### 5.4 Prototyping / Iterative

**Typical contexts:** RBI Regulatory Sandbox cohorts, limited-user proof-of-concept pilots, Account Aggregator data-visualization and analytics experiments.

**Guardrails:** defined scope, user caps, exit criteria, and data protection (consent under the DPDP Act) even during a pilot.

**Graduation:** a successful prototype should be rebuilt or hardened under Agile-DevSecOps or V-Model before production scale. It should not be shipped as is.

## 6. Hybrid and Edge Cases

- **Core banking modernization:** ledger and interest engines follow V-Model. Surrounding channels and APIs follow Agile-DevSecOps.
- **UPI infrastructure:** transaction-processing backends follow V-Model style. The customer UI follows Agile.
- **Trading platforms:** the retail front end follows Agile. The order gateway and algo components follow V-Model because of exchange and SEBI audit scope.
- **Digital lending:** the UX follows Agile. Credit decisioning and disbursal ledger changes go through gated release. Lending service provider governance applies.
- **AI/ML components (credit scoring, fraud):** add model validation, drift and bias monitoring and explainability. See the RBI FREE-AI framework.
- **Emergency security patches:** an expedited change path is acceptable if documented and approved. Audit logging is never bypassed.
- **Supervisory letters and audit findings:** these override general guidance.

## 7. Worked Examples

**RTGS participant gateway:** irreversible high-value settlement, systemic impact, per-release attestation, stable requirements. Recommended: V-Model with DevSecOps gates. Evidence: traceability matrix, integration and DR tests, independent VAPT.

**UPI-enabled mobile banking feature team:** frequent change, reversible releases, sensitive data. Recommended: Agile-DevSecOps with SAST, SCA, secret scanning, app security testing, maker-checker deploy approval, and scheduled NPCI or bank certification steps.

**HSM firmware build:** frozen specification, no post-release patching. Recommended: Waterfall / staged-gate, with the management software built separately using Agile.

**Is Agile prohibited for core banking?** No. No regulator prohibits Agile. Ledger-critical modules usually adopt gated, V-Model-style verification because of the assurance burden, but Agile increments are acceptable if traceability, testing and approval evidence are produced.

## 8. Corrections to Common Misstatements

- "The regulator mandates the V-Model": incorrect. Regulators mandate controls and evidence, not a model.
- "Pure Agile always fails in banking": overstated. Agile with automated gates and a proper release approval process is widely used for customer-facing systems.
- "SEBI CSCRF is a 2023 circular": incorrect. CSCRF was issued in August 2024.
- "RBI Master Direction on Cyber Security Framework": there is no single document by that name. The relevant sources are the 2016 Cyber Security Framework circular, the 2021 Digital Payment Security Controls direction and the 2023 IT Governance direction.
- "HSMs need FIPS 140-2 Level 3": FIPS 140-3 is the current standard, and 140-2 certificates are being phased out.
