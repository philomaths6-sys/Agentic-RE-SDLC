# guardrails/input_filter.py
"""Zero-Trust Input Guardrails with Regex-based Detection, Verhoeff Aadhaar Verification,
PAN Masking, Context-Aware Security Scanning, and Prompt Injection Defense.

Key Capabilities:
1. PII Detection & Redaction:
   - Aadhaar 12-digit number validation using official Verhoeff checksum algorithm.
   - Income Tax PAN validation (5 uppercase letters + 4 digits + 1 uppercase letter).
   - Secret/Password credential masking.
2. Context-Aware Security Check:
   - Word-boundary matching with defensive security context awareness (e.g., "brute force attack prevention"
     and "hackathon" are allowed, while adversarial system execution like "rm -rf" or "drop database" is blocked).
3. Prompt Injection Classifier:
   - Detects jailbreaks, instruction override attempts ("ignore previous instructions"), roleplay escapes,
     and special token delimiter injections.
4. Affirmative Financial Domain Scope Validation:
   - Evaluates positive alignment with Indian Regulated Financial Infrastructure (RBI, SEBI, NPCI, UIDAI, DPDP).
"""

import re
from typing import Tuple, List, Dict, Any


# ---------------------------------------------------------------------------
# 1. VERHOEFF CHECKSUM ALGORITHM (Official Aadhaar Validation)
# ---------------------------------------------------------------------------
# Multiplication table d
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Permutation table p
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

# Inverse table
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff(num_str: str) -> bool:
    """Validate a numeric string using the Verhoeff algorithm."""
    clean = re.sub(r"\D", "", num_str)
    if not clean or len(clean) != 12:
        return False
    # Verhoeff checksum calculation
    c = 0
    reversed_digits = [int(x) for x in reversed(clean)]
    for i, digit in enumerate(reversed_digits):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][digit]]
    return c == 0


def validate_pan(pan_str: str) -> bool:
    """Validate Indian Income Tax PAN format: 5 letters + 4 digits + 1 letter."""
    clean = pan_str.strip().upper()
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", clean))


# ---------------------------------------------------------------------------
# 2. PII DETECTION & REDACTION ENGINE
# ---------------------------------------------------------------------------
# Regex patterns
_AADHAAR_REGEX = re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b")
_PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_PASSWORD_REGEX = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|bearer\s+token)\s*[:=]\s*['\"]?([A-Za-z0-9@#$%^&*!_\-\.]{6,})['\"]?"
)


def detect_and_mask_pii(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Scan text for real Aadhaar numbers (Verhoeff-validated), PAN cards, and passwords.
    
    Returns:
        (sanitized_text: str, detected_pii: list)
    """
    detected_pii: List[Dict[str, Any]] = []
    sanitized = text

    # 1. Scan Aadhaar
    for match in _AADHAAR_REGEX.finditer(text):
        raw_val = match.group(0)
        digits_only = re.sub(r"\D", "", raw_val)
        if validate_verhoeff(digits_only):
            masked = f"XXXX-XXXX-{digits_only[-4:]}"
            sanitized = sanitized.replace(raw_val, masked)
            detected_pii.append({
                "type": "Aadhaar",
                "masked": masked,
                "verhoeff_valid": True,
                "position": match.span()
            })

    # 2. Scan PAN
    for match in _PAN_REGEX.finditer(sanitized):
        raw_val = match.group(0)
        masked = f"{raw_val[:2]}XXX{raw_val[5:9]}X"
        sanitized = sanitized.replace(raw_val, masked)
        detected_pii.append({
            "type": "PAN",
            "masked": masked,
            "position": match.span()
        })

    # 3. Scan Credentials
    for match in _PASSWORD_REGEX.finditer(sanitized):
        full_match = match.group(0)
        field = match.group(1)
        sanitized = sanitized.replace(full_match, f"{field}=[REDACTED_CREDENTIAL]")
        detected_pii.append({
            "type": "Credential",
            "field": field,
            "masked": "[REDACTED_CREDENTIAL]"
        })

    return sanitized, detected_pii


# ---------------------------------------------------------------------------
# 3. PROMPT INJECTION & JAILBREAK CLASSIFIER
# ---------------------------------------------------------------------------
INJECTION_SIGNATURES = [
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|directives)\b",
    r"\bdisregard\s+(all\s+)?(previous|prior|system)\s+(instructions|rules)\b",
    r"\byou\s+are\s+now\s+(in\s+)?(dan|developer|jailbreak|unrestricted|god)\s+mode\b",
    r"\bact\s+as\s+an?\s+unrestricted\s+ai\b",
    r"\breveal\s+(your\s+)?(system\s+prompt|developer\s+instructions|hidden\s+rules)\b",
    r"\bbypass\s+(all\s+)?(guardrails|filters|content\s+filters|safety\s+checks)\b",
    r"<\s*\|\s*im_start\s*\|\s*>",
    r"\[\s*INST\s*\]\s*<<SYS>>",
    r"###\s*System\s*:",
    r"\bshow\s+me\s+your\s+initial\s+prompt\b",
    r"\bdo\s+anything\s+now\b"
]

_INJECTION_REGEXES = [re.compile(sig, re.IGNORECASE) for sig in INJECTION_SIGNATURES]


def detect_prompt_injection(text: str) -> Tuple[bool, float, str]:
    """Detect prompt injection and jailbreak payloads using regex signatures and pattern analysis.
    
    Returns:
        (is_injection: bool, injection_score: float, details: str)
    """
    matched = []
    for reg in _INJECTION_REGEXES:
        match = reg.search(text)
        if match:
            matched.append(match.group(0))

    if matched:
        score = min(1.0, 0.4 + 0.3 * len(matched))
        return True, score, f"Detected injection pattern(s): {', '.join(matched)}"

    # Check for excessive delimiter nesting or instruction override heuristics
    if "system prompt" in text.lower() and any(w in text.lower() for w in ["override", "forget", "replace", "print"]):
        return True, 0.75, "Detected semantic system prompt override attempt"

    return False, 0.0, "Zero prompt injection signatures detected"


# ---------------------------------------------------------------------------
# 4. CONTEXT-AWARE SECURITY GUARDRAIL (Word-Boundary + Context Exemption)
# ---------------------------------------------------------------------------
# Truly malicious / destructive administrative commands (Always blocked)
DISALLOWED_SYSTEM_COMMANDS = [
    r"\bdrop\s+(database|table|schema|user)\b",
    r"\bshutdown\s+(server|system|now)\b",
    r"\brun\s+sudo\b",
    r"\bsudo\s+rm\s+-[rRfF]+\b",
    r"\brm\s+-[rRfF]+\s+[/~*]\b",
    r"\bchmod\s+777\s+[/~*]\b",
    r"\bdelete\s+from\s+[a-zA-Z_]+\s+where\s+1\s*=\s*1\b",
    r"\bformat\s+c:\b",
    r"\bmkfs(\.ext\d)?\b",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",  # Fork bomb
]

_SYS_CMD_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in DISALLOWED_SYSTEM_COMMANDS]

# Words that need context analysis
_SECURITY_TERMS = [r"\bhack\b", r"\battack\b", r"\bexploit\b", r"\bmalware\b", r"\bvirus\b", r"\bjailbreak\b"]
_DEFENSIVE_CONTEXT_REGEX = re.compile(
    r"(?i)\b(prevent|preventing|prevention|mitigate|mitigating|mitigation|defend|defense|protect|protection|"
    r"detect|detection|anti|shield|filter|against|security|resilience|vulnerability|assessment|audit|"
    r"brute\s*force|ddos|sql\s*injection|cross\s*site|zero\s*day|patch|hardening|secure|hackathon)\b"
)


def is_safe(text: str) -> bool:
    """Return True if text does not contain destructive system commands or malicious exploits.
    
    Legitimate security engineering specifications (e.g. 'anti-DDoS attack prevention',
    'exploit mitigation', 'hackathon project', 'malware scanning') are explicitly permitted.
    """
    lowered = text.lower()

    # 1. Block destructive system commands unconditionally
    for reg in _SYS_CMD_REGEXES:
        if reg.search(text):
            return False

    # 2. Block direct prompt injection attempts
    is_inj, _, _ = detect_prompt_injection(text)
    if is_inj:
        return False

    # 3. Contextual inspection for security terminology
    for term in _SECURITY_TERMS:
        match = re.search(term, lowered)
        if match:
            # Check if text describes a legitimate security requirement / defensive control / hackathon
            if "hackathon" in lowered:
                continue
            if _DEFENSIVE_CONTEXT_REGEX.search(lowered):
                continue
            # Standalone malicious intent without defensive context
            if re.search(r"\b(how\s+to|help\s+me|write\s+a)\s+(hack|exploit|attack|malware|virus)\b", lowered):
                return False

    return True


# ---------------------------------------------------------------------------
# 5. FINANCIAL DOMAIN TAXONOMY & AFFIRMATIVE SCOPE GATE
# ---------------------------------------------------------------------------
FINANCIAL_SUBDOMAINS = {
    "Payments, Switches & Mandates": [
        "payment", "payments", "upi", "autopay", "mandate", "wallet", "pos",
        "debit card", "credit card", "payment card", "rupay", "visa", "mastercard",
        "qr", "merchant", "bbps", "tokenization", "switch", "settlement", "clearing",
        "gateway", "remittance", "transaction", "checkout", "interbank", "coft", "payout"
    ],
    "Banking & Core Ledgers": [
        "bank", "banking", "cbs", "core banking", "account", "ledger", "deposit", "withdrawal",
        "neft", "rtgs", "imps", "nach", "enach", "atm", "cash", "vault", "cassette",
        "passbook", "statement", "overdraft", "treasury", "swift", "reconciliation", "double-entry"
    ],
    "Lending, Credit & Underwriting": [
        "loan", "lending", "borrower", "lender", "nbfc", "cibil", "crif",
        "experian", "equifax", "underwriting", "disbursal", "repayment", "emi",
        "kfs", "apr", "interest", "dbt", "microfinance", "shg", "collateral",
        "mortgage", "debt", "credit risk", "credit scoring", "credit"
    ],
    "Capital Markets, Trading & Wealth": [
        "trading", "trade", "algo trading", "algo-trading", "algorithmic trading", "trading algo", "broker", "stock", "stocks",
        "equity", "equities", "share", "shares", "demat", "depository", "nsdl",
        "cdsl", "nse", "bse", "sebi", "derivative", "derivatives", "futures",
        "options", "fix protocol", "colocation", "order routing", "rms", "circuit breaker",
        "portfolio", "mutual fund", "sip", "amc", "nav", "hft"
    ],
    "Insurance & Actuarial Systems": [
        "insurance", "policy", "premium", "claim", "claims", "underwrite", "tpa",
        "irdai", "annuity", "coverage", "actuarial", "cashless", "indemnity",
        "reimbursement", "solvency", "actuary"
    ],
    "Financial Identity, KYC & Statutory Governance": [
        "kyc", "v-cip", "aadhaar", "uidai", "pan", "c-kyc", "ckyc", "data vault",
        "aml", "pmla", "str", "ctr", "fiu", "fiu-ind", "cscrf", "fraud",
        "anti-money laundering", "sanction", "audit trail", "rbi", "npci",
        "dpdp", "data localization", "data residency", "cert-in", "ssdf", "vapt"
    ]
}

FINANCIAL_DOMAIN_KEYWORDS = set(
    kw for sublist in FINANCIAL_SUBDOMAINS.values() for kw in sublist
)


def classify_financial_domain(text: str) -> list:
    """Inspect text and return all affirmative financial domain tags that match."""
    lowered = text.lower()
    matched_tags = []
    for domain, kws in FINANCIAL_SUBDOMAINS.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", lowered) for kw in kws):
            matched_tags.append(domain)
    return matched_tags


OUT_OF_DOMAIN_CATEGORIES = [
    {
        "category": "Healthcare & Clinical Management",
        "patterns": [
            r"\bhealth\s*sector\b", r"\bhealth\s*care\b", r"\bhealthcare\b",
            r"\bhospital\b", r"\bclinic\b", r"\bclinical\b", r"\bpatient\b",
            r"\bdoctor\b", r"\bnurse\b", r"\bward\b", r"\bbed\s*management\b",
            r"\bradiology\b", r"\bpathology\b", r"\behr\b", r"\bemr\b",
            r"\bprescription\b", r"\bsurgery\b", r"\bicu\b", r"\bpharmacy\b",
            r"\bmedical\b", r"\btriage\b", r"\bappointment\s*booking\b",
            r"\bhealth\s*app\b", r"\btelemedicine\b"
        ],
        "allowed_exceptions": [
            "insurance", "claim", "claims", "tpa", "billing", "payment",
            "payments", "reimbursement", "premium", "policy", "gateway"
        ],
        "clarification": "If you intended to build an IRDAI-regulated Health Insurance & TPA Claims settlement platform or a Hospital Payment Processing Gateway, please include those financial terms in your query."
    },
    {
        "category": "Gaming & Entertainment",
        "patterns": [
            r"\bvideo\s*game\b", r"\bgame\s*engine\b", r"\brpg\b", r"\bmultiplayer\s*game\b",
            r"\besports\b", r"\bmovie\s*streaming\b", r"\bmusic\s*streaming\b", r"\bpodcast\s*player\b"
        ],
        "allowed_exceptions": ["in-game purchase", "payment gateway", "wallet", "checkout", "microtransaction"],
        "clarification": "General gaming engines, entertainment streaming, and esports mechanics are outside the financial engineering domain."
    },
    {
        "category": "Food, Delivery & Culinary",
        "patterns": [
            r"\brecipe\b", r"\bcooking\b", r"\bmeal\s*planner\b", r"\brestaurant\b",
            r"\bchef\b", r"\bingredient\b", r"\bdiet\s*tracker\b", r"\bpizza\b",
            r"\bfood\s*deliver(y|ing)?\b", r"\bfood\s*order(ing)?\b", r"\bkitchen\b",
            r"\bmenu\s*item\b", r"\bbakery\b", r"\bgrocery\s*deliver(y|ing)?\b",
            r"\bswiggy\b", r"\bzomato\b", r"\bdomino\'?s?\b", r"\bcatering\b",
            r"\btakeaway\b", r"\btakeout\b"
        ],
        "allowed_exceptions": ["payment", "billing", "pos", "checkout", "order payment", "payment gateway", "settlement", "refund", "wallet", "upi"],
        "clarification": "General food delivery, pizza ordering, culinary, and kitchen management systems are outside the financial engineering domain."
    },
    {
        "category": "Ride Hailing & Transport Logistics",
        "patterns": [
            r"\bcab\s*booking\b", r"\btaxi\s*app\b", r"\bride\s*sharing\b", r"\bbike\s*taxi\b",
            r"\buber\b", r"\bola\b", r"\brapido\b", r"\bcarpool\b", r"\bfleet\s*dispatch\b"
        ],
        "allowed_exceptions": ["fare payment", "driver payout", "wallet", "upi", "card tokenization", "payment gateway", "toll payment"],
        "clarification": "Ride hailing, vehicle dispatch, and driver GPS navigation are outside the financial domain."
    },
    {
        "category": "Hotels & Hospitality Accommodation",
        "patterns": [
            r"\bhotel\s*booking\b", r"\broom\s*reservation\b", r"\bhostel\s*booking\b",
            r"\bairbnb\b", r"\bhotel\s*management\b", r"\bguest\s*check-?in\b"
        ],
        "allowed_exceptions": ["booking payment", "advance deposit", "card tokenization", "refund switch", "payment"],
        "clarification": "Hotel room inventory, housekeeping, and front-desk booking engines are outside the financial domain."
    },
    {
        "category": "General E-Commerce & Retail Goods",
        "patterns": [
            r"\bclothing\s*store\b", r"\bfashion\s*app\b", r"\bfurniture\s*store\b",
            r"\bshoe\s*store\b", r"\bapparel\b", r"\bjewelry\s*shop\b"
        ],
        "allowed_exceptions": ["payment gateway", "card-on-file", "tokenization", "bnpl", "checkout", "emi", "refund", "upi"],
        "clarification": "General apparel, furniture, and retail merchandise catalogs are outside the financial domain."
    },
    {
        "category": "Academic & School Administration",
        "patterns": [
            r"\bstudent\s*attendance\b", r"\bteacher\s*grading\b", r"\breport\s*card\b",
            r"\bschool\s*management\b", r"\blms\s*quiz\b", r"\bhomework\b", r"\bclassroom\b"
        ],
        "allowed_exceptions": ["fee payment", "education loan", "scholarship", "disbursal", "payment", "tuition fee"],
        "clarification": "Academic classroom grading, attendance rosters, and curriculum tracking are outside the financial domain."
    },
    {
        "category": "Automotive & Mechanical Diagnostics",
        "patterns": [
            r"\bcar\s*engine\b", r"\bvehicle\s*diagnostic\b", r"\bflight\s*simulator\b",
            r"\bdrone\s*navigation\b", r"\btraffic\s*light\b", r"\bautomotive\s*mechanic\b"
        ],
        "allowed_exceptions": ["motor insurance", "fastag", "toll payment", "vehicle loan", "auto finance"],
        "clarification": "Mechanical diagnostics, vehicle telematics, and avionics are outside the financial domain."
    },
    {
        "category": "Social Networking & Dating",
        "patterns": [
            r"\bdating\s*app\b", r"\bmatchmaking\b", r"\btinder\b", r"\bphoto\s*sharing\s*filter\b",
            r"\bsocial\s*media\s*feed\b", r"\bchat\s*room\b"
        ],
        "allowed_exceptions": ["creator payout", "tip payment", "wallet", "payment"],
        "clarification": "Social networking feeds, photo filters, and dating mechanics are outside the financial domain."
    }
]


def check_domain_scope(text: str) -> Tuple[bool, str, str]:
    """Affirmative Positive Domain Tagging & Scope Evaluation."""
    lowered = text.lower()
    matched_tags = classify_financial_domain(text)

    # 1. Evaluate explicit out-of-domain categories
    for cat in OUT_OF_DOMAIN_CATEGORIES:
        has_cat_match = any(re.search(pat, lowered) for pat in cat["patterns"])
        if has_cat_match:
            has_exception = any(exc in lowered for exc in cat["allowed_exceptions"])
            if has_exception or matched_tags:
                primary = matched_tags[0] if matched_tags else "Regulated FinTech Component"
                return True, f"{primary} ({cat['category']} integration)", ""
            return False, cat["category"], cat["clarification"]

    # 2. Positive Affirmative Domain Check
    if matched_tags:
        return True, matched_tags[0], ""

    clarification = (
        "The Supervisor Agent detected 0 financial domain tags in this request. "
        "The platform specializes strictly in Indian Regulated Financial Infrastructure "
        "(RBI, SEBI, NPCI, UIDAI, IRDAI, and DPDP Act 2023). "
        "Please specify the digital payment, banking ledger, loan origination, trading, "
        "insurance claim, or KYC compliance scope of your project."
    )
    return False, "Non-Financial / General Domain", clarification


# ---------------------------------------------------------------------------
# 6. CASUAL CONVERSATION & GREETING CLASSIFIER
# ---------------------------------------------------------------------------
GREETINGS_SET = {
    "hi", "hello", "hey", "how are you", "how r u", "who are you", "who r u",
    "what can you do", "good morning", "good afternoon", "good evening",
    "help", "test", "sup", "whats up", "what is this", "explain yourself",
    "tell me about yourself", "hi there", "hello there", "thanks", "thank you"
}


def is_conversational(text: str) -> bool:
    """Detect whether input is casual conversation, greeting, or help question rather than a project requirement."""
    cleaned = text.strip().lower().rstrip("?!.")
    if cleaned in GREETINGS_SET:
        return True

    # Common greeting prefixes (e.g. "hello assistant", "hey there")
    if any(cleaned.startswith(g + " ") for g in ["hi", "hello", "hey", "good morning", "good evening"]):
        return True

    return False


def get_conversational_response(user_text: str) -> str:
    """Return a natural, conversational response for casual questions."""
    text = user_text.lower()
    if "how are you" in text:
        return (
            "I'm doing well, thank you for asking! 😊\n\n"
            "I am your **Agentic RE-SDLC Advisor**, an AI system powered by local models for Indian Banking, FinTech, and Regulated Financial Enterprises.\n\n"
            "You can describe a software system you want to build (e.g. *'Design an algorithmic trading platform for NSE equity derivatives with FIX protocol'* or *'Create a Microfinance loan origination app for rural farmers'*), or upload an SRS draft (.docx, .pdf) in the sidebar. I will dynamically generate a comprehensive 30-point Software Requirement Specification (SRS) with regulatory compliance mapping and predict the optimal SDLC model!\n\n"
            "What financial project or requirement would you like to build today?"
        )
    elif any(g in text for g in ["hi", "hello", "hey", "good morning", "good evening"]):
        return (
            "Hello! 👋 Welcome to the **Agentic RE-SDLC Advisor**.\n\n"
            "I am ready to help you engineer detailed, production-grade financial software requirements compliant with **RBI, NPCI, UIDAI, DPDP Act 2023, and SEBI** regulations.\n\n"
            "Tell me about your project or upload your requirement document, and I will generate:\n"
            "- **30 Detailed Modular Requirements** across all 6 Architectural Pillars\n"
            "- **MoSCoW Prioritization & Statutory Citations**\n"
            "- **Downloadable Executive SRS PDF Report**\n"
            "- **Fine-Tuned SDLC Recommendation & Rationale**"
        )
    else:
        return (
            "I am your **Agentic RE-SDLC Advisor** for Indian Banking & Financial Systems.\n\n"
            "To generate a complete, structured Software Requirements Specification (SRS) and fine-tuned SDLC recommendation:\n"
            "1. Type your project scope or requirement in the chat.\n"
            "2. Or upload a project document (.docx, .pdf, .txt) using the sidebar uploader.\n\n"
            "What would you like to build or analyze?"
        )


# ---------------------------------------------------------------------------
# 7. UNIFIED INPUT SCANNER FOR SUPERVISOR AGENT
# ---------------------------------------------------------------------------
def scan_input_payload(text: str) -> Dict[str, Any]:
    """Execute complete multi-layered Zero-Trust input perimeter scan.
    
    Returns structured audit record with masked text and dynamic metrics.
    """
    # 0. Conversational Intent Check (Casual Greetings & Inquiries)
    if is_conversational(text):
        return {
            "passed": True,
            "is_safe": True,
            "is_injection": False,
            "injection_score": 0.0,
            "injection_details": "",
            "in_domain": True,
            "domain_tag": "Conversational Greeting",
            "domain_reason": "",
            "sanitized_text": text,
            "pii_detected": [],
            "threat_assessment": "Conversational Greeting / Inquiry",
            "is_conversational": True,
            "conversational_response": get_conversational_response(text)
        }

    # 1. PII Scan & Redaction
    sanitized_text, pii_records = detect_and_mask_pii(text)

    # 2. Prompt Injection Classification
    is_inj, inj_score, inj_details = detect_prompt_injection(sanitized_text)

    # 3. Security Command & Exploit Check
    safe_status = is_safe(sanitized_text)

    # 4. Domain Boundary Check
    in_domain, domain_tag, domain_reason = check_domain_scope(sanitized_text)

    passed_all = safe_status and (not is_inj) and in_domain

    threat_summary = []
    if pii_records:
        threat_summary.append(f"{len(pii_records)} sensitive PII entity(s) masked")
    else:
        threat_summary.append("Zero PII detected")

    if is_inj:
        threat_summary.append(f"Prompt injection detected (risk score {inj_score:.2f})")
    else:
        threat_summary.append(f"Zero prompt injection (score {inj_score:.2f})")

    return {
        "passed": passed_all,
        "is_safe": safe_status,
        "is_injection": is_inj,
        "injection_score": inj_score,
        "injection_details": inj_details,
        "in_domain": in_domain,
        "domain_tag": domain_tag,
        "domain_reason": domain_reason,
        "sanitized_text": sanitized_text,
        "pii_detected": pii_records,
        "threat_assessment": " | ".join(threat_summary),
        "is_conversational": False,
        "conversational_response": ""
    }
