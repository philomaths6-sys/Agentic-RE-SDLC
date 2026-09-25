# guardrails/input_filter.py
"""Input guardrails for safety and domain boundary enforcement.

Enforces:
1. Zero-Trust Security Guardrails: Prohibits prompt injection, adversarial commands, and destructive actions.
2. Domain Scope Guardrails: Ensures user requests fall within Indian Regulated Financial Infrastructure
   (RBI, SEBI, NPCI, IRDAI, PMLA, DPDP Act 2023), preventing model hallucination or force-fitting out-of-domain queries.
"""

import re
from typing import Tuple

# Disallowed attack patterns and adversarial commands
DISALLOWED_INPUT_PHRASES = {
    "drop database",
    "shutdown server",
    "run sudo",
    "delete all",
    "rm -rf",
    "hack",
    "attack",
    "malware",
    "virus",
    "exploit",
    "jailbreak",
}

# Recognized Financial & FinTech Domain Sub-domains and Keywords for Positive Affirmative Tagging
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
        "trading", "trade", "algo", "algorithmic", "broker", "stock", "stocks",
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

# Flattened set for backward compatibility
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


# Out-of-domain categories and detection rules
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
        "clarification": "General food delivery, pizza ordering, culinary, and kitchen management systems are outside the financial engineering domain. If you are building the digital payment checkout or merchant settlement switch for this service, please specify the financial requirements."
    },
    {
        "category": "Ride Hailing & Transport Logistics",
        "patterns": [
            r"\bcab\s*booking\b", r"\btaxi\s*app\b", r"\bride\s*sharing\b", r"\bbike\s*taxi\b",
            r"\buber\b", r"\bola\b", r"\brapido\b", r"\bcarpool\b", r"\bfleet\s*dispatch\b"
        ],
        "allowed_exceptions": ["fare payment", "driver payout", "wallet", "upi", "card tokenization", "payment gateway", "toll payment"],
        "clarification": "Ride hailing, vehicle dispatch, and driver GPS navigation are outside the financial domain. Include payment gateway or driver payout details if designing the financial subsystem."
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
        "clarification": "General apparel, furniture, and retail merchandise catalogs are outside the financial domain. Specify payment gateway or checkout tokenization if designing the financial component."
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


def is_safe(text: str) -> bool:
    """Return True if text does not contain disallowed security attack phrases."""
    lowered = text.lower()
    return not any(phrase in lowered for phrase in DISALLOWED_INPUT_PHRASES)


def check_domain_scope(text: str) -> Tuple[bool, str, str]:
    """Affirmative Positive Domain Tagging & Scope Evaluation.
    
    The Supervisor Agent evaluates whether the user prompt affirmatively carries
    at least one verified financial engineering domain tag (Banking, Payments, Lending,
    Trading, Insurance, KYC/Compliance).
    
    Returns:
        (is_in_domain: bool, detected_category_or_tag: str, clarification: str)
    """
    lowered = text.lower()
    matched_tags = classify_financial_domain(text)

    # 1. Evaluate explicit out-of-domain categories
    for cat in OUT_OF_DOMAIN_CATEGORIES:
        has_cat_match = any(re.search(pat, lowered) for pat in cat["patterns"])
        if has_cat_match:
            # Check if an explicit financial exception is present
            has_exception = any(exc in lowered for exc in cat["allowed_exceptions"])
            if has_exception or matched_tags:
                primary = matched_tags[0] if matched_tags else "Regulated FinTech Component"
                return True, f"{primary} ({cat['category']} integration)", ""
            return False, cat["category"], cat["clarification"]

    # 2. Positive Affirmative Domain Check (Zero-Trust Gating)
    # The query must affirmatively carry at least one financial sub-domain tag.
    if matched_tags:
        return True, matched_tags[0], ""

    # 3. Intercept non-financial / untagged prompts
    clarification = (
        "The Supervisor Agent detected 0 financial domain tags in this request. "
        "The platform specializes strictly in Indian Regulated Financial Infrastructure "
        "(RBI, SEBI, NPCI, UIDAI, IRDAI, and DPDP Act 2023). "
        "Please specify the digital payment, banking ledger, loan origination, trading, "
        "insurance claim, or KYC compliance scope of your project."
    )
    return False, "Non-Financial / General Domain", clarification
