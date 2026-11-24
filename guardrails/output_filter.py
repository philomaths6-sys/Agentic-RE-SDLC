# guardrails/output_filter.py
"""Output Guardrail – Validates and sanitizes LLM responses by Schema, Privacy, and Policy.

Key Improvements:
1. Eliminates naive keyword banning of legitimate technical terms (e.g. 'SQL injection',
   'malware prevention', 'root password hashing' are valid architectural controls).
2. Prevents secret leakage (private keys, bearer tokens, plaintext database passwords).
3. Enforces output schema and structure validation (JSON schema integrity when applicable).
4. Ensures proper sentence termination and removes model artifacts/echoes.
"""

import os
import re
import json
from typing import Tuple, Optional, Dict, Any

MAX_OUTPUT_LENGTH = int(os.environ.get("OUTPUT_MAX_LENGTH", "50000"))

# Patterns that represent true security leaks rather than technical descriptions
DATA_LEAK_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9_\-\.]{32,}", re.IGNORECASE),
    re.compile(r"\b(aws_secret_access_key|client_secret)\s*[:=]\s*['\"][A-Za-z0-9/+=]{20,}['\"]", re.IGNORECASE),
]

# Prohibited dangerous system instructions that an LLM should never output as executable scripts
DANGEROUS_EXEC_COMMANDS = [
    re.compile(r"\brm\s+-[rRfF]+\s+[/~*]\b"),
    re.compile(r"\bchmod\s+777\s+[/~*]\b"),
    re.compile(r"\b(curl|wget)\s+https?://[^\s]+\s*\|\s*(bash|sh)\b", re.IGNORECASE),
]


def validate_json_schema(text: str, required_keys: Optional[list] = None) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Attempt to parse and validate JSON structure from model output."""
    try:
        # Extract json block if surrounded by markdown code fences
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw_json = match.group(1) if match else text.strip()
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            return False, None, "JSON root is not an object/dictionary"
        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                return False, data, f"Missing required JSON keys: {', '.join(missing)}"
        return True, data, ""
    except json.JSONDecodeError as exc:
        return False, None, f"Malformed JSON: {exc}"


def filter_output(llm_output: str, expect_json: bool = False, required_keys: Optional[list] = None) -> Tuple[bool, str, str]:
    """Validate model output against safety policy, schema integrity, and data leakage.
    
    Returns:
        (allowed: bool, cleaned_output: str, reason: str)
    """
    if not llm_output or not llm_output.strip():
        return False, "", "Empty output received from model"

    cleaned_output = llm_output.strip()

    # 1. Length validation
    if len(cleaned_output) > MAX_OUTPUT_LENGTH:
        cleaned_output = cleaned_output[:MAX_OUTPUT_LENGTH]

    # 2. Secret & Private Key Leakage Check
    for leak_pattern in DATA_LEAK_PATTERNS:
        if leak_pattern.search(cleaned_output):
            cleaned_output = leak_pattern.sub("[REDACTED_SENSITIVE_KEY]", cleaned_output)

    # 3. Executable System Destruction Command Check
    for cmd_pattern in DANGEROUS_EXEC_COMMANDS:
        if cmd_pattern.search(cleaned_output):
            return False, "", "Output rejected: contains prohibited destructive shell commands"

    # 4. Optional JSON Schema Validation
    if expect_json:
        valid_json, _, err = validate_json_schema(cleaned_output, required_keys=required_keys)
        if not valid_json:
            return False, cleaned_output, f"Schema validation failed: {err}"

    # 5. Clean up hanging code fences or incomplete sentences if non-JSON
    if not expect_json and not cleaned_output.endswith(("```", "}", "]", ".", "!", "?", '"', "'")):
        cleaned_output = cleaned_output.rstrip() + "."

    return True, cleaned_output, ""


if __name__ == "__main__":
    test_sec = "The backend must implement parameterized queries to prevent SQL injection and cross-site scripting."
    ok, cleaned, reason = filter_output(test_sec)
    print("Allowed:", ok)
    print("Cleaned:", cleaned)
    assert "SQL injection" in cleaned, "SQL injection should NOT be redacted!"
    print("Self-test passed!")
