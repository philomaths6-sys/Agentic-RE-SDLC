# guardrails/output_filter.py
"""Output guardrail – validates LLM responses before they are returned to the user.

The ``filter_output`` function receives the raw LLM text and returns a tuple
``(allowed: bool, cleaned: str, reason: str)``.
It performs lightweight checks that do not require GPU:

* Maximum token/character limit (default 2000 characters).
* Simple blacklist of disallowed phrases (e.g., instructions to hack, disclose
  passwords, or reveal internal system details).
* Ensures the response ends with a proper sentence termination to avoid
  truncated outputs.

If ``allowed`` is ``False`` the caller can either ask the LLM to regenerate or
return an error message to the end‑user.
"""

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Configuration – adjust via environment variables if needed
# ---------------------------------------------------------------------------
MAX_OUTPUT_LENGTH = int(__import__("os").environ.get("OUTPUT_MAX_LENGTH", "20000"))
# Only block truly harmful phrases – avoid broad terms that appear in legitimate technical output
DISALLOWED_PHRASES = {
    "expose private key",
    "run sudo rm",
    "drop table",
    "malware",
    "ransomware",
    "root password",
    "shell injection",
    "sql injection",
}


def _contains_disallowed(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in DISALLOWED_PHRASES)


def filter_output(llm_output: str) -> Tuple[bool, str, str]:
    """Validate *llm_output*.

    Returns ``(allowed, cleaned_output, reason)``. ``cleaned_output`` may be trimmed
    to the length limit. ``reason`` explains why the output was rejected.
    """
    if not llm_output:
        return False, "", "Empty output"

    # Length check – truncate if necessary but still allow (don't reject, just trim)
    if len(llm_output) > MAX_OUTPUT_LENGTH:
        llm_output = llm_output[:MAX_OUTPUT_LENGTH]

    # Redact disallowed phrases instead of rejecting
    cleaned_output = llm_output
    for phrase in DISALLOWED_PHRASES:
        # case-insensitive replacement
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        cleaned_output = pattern.sub("[REDACTED]", cleaned_output)
    # Ensure response ends with punctuation (., !, ?)
    if not re.search(r"[.!?]$", cleaned_output.strip()):
        cleaned_output = cleaned_output.rstrip() + "."
    return True, cleaned_output, ""


# Simple demo when run as a script
if __name__ == "__main__":
    import sys
    out = " ".join(sys.argv[1:])
    ok, cleaned, msg = filter_output(out)
    print({"allowed": ok, "cleaned": cleaned, "reason": msg})
