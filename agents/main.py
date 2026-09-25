# agents/main.py
"""Entry point for the requirement‑generation agent.

Usage:
    python -m agents.main "<natural language request>"

The agent:
1. Validates the user request with the input guard (Llama‑Guard, CPU).
2. Retrieves relevant chunks from Qdrant.
3. Constructs a prompt that includes the retrieved context.
4. Generates a requirement using the base LLM (Llama‑3.2‑3B‑Instruct, GPU).
5. Validates the LLM output with the output guard.
6. Prints the safe requirement or an error message.
"""

import os
import torch
import sys
from typing import List

# Guardrails
from guardrails.input_filter import is_safe as input_is_safe
from guardrails.output_filter import filter_output

# Retrieval
from rag.retriever import retrieve

# LLM (GPU‑enabled)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def _load_llm():
    """Load the local 4-bit quantized Qwen-2.5-3B model."""
    model_name = os.getenv("LLM_MODEL", "unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
    except Exception as e:
        print(f"[WARN] Failed to load LLM model '{model_name}': {e}. Falling back to 'gpt2'.")
        fallback_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(fallback_name)
        model = AutoModelForCausalLM.from_pretrained(fallback_name, device_map="auto")

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )
    return gen_pipe


def _build_prompt(query: str, contexts: List[dict]) -> str:
    """Create a prompt that combines retrieved context with the user query.

    The prompt instructs the LLM to produce a concise requirement.
    """
    context_texts = [c.get("text", "") for c in contexts]
    context_block = "\n---\n".join(context_texts)
    prompt = (
        "You are an expert requirements engineer for financial services. "
        "Using the following regulatory context, draft a precise requirement that addresses the user's request.\n\n"
        f"Context:\n{context_block}\n\n"
        f"User request: {query}\n\n"
        "Requirement: (Do not include any personal data such as passwords, email addresses, or telephone numbers)"
    )
    return prompt


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m agents.main \"<request>\"")
        sys.exit(1)
    user_query = " ".join(sys.argv[1:])

    # 1️⃣ Input guard
    if not input_is_safe(user_query):
        print("[ERROR] Input rejected by guardrails.")
        sys.exit(1)

    # 2️⃣ Retrieval
    retrieved = retrieve(user_query, top_k=5)

    # 3️⃣ Prompt construction
    prompt = _build_prompt(user_query, retrieved)

    # 4️⃣ LLM generation
    llm = _load_llm()
    generated = llm(prompt)[0]["generated_text"]

    # 5️⃣ Output guard
    allowed, cleaned, reason = filter_output(generated)
    if not allowed:
        print(f"[ERROR] Output rejected by guardrails: {reason}")
        sys.exit(1)

    # 6️⃣ Show result
    print("--- Requirement ---")
    print(cleaned)

    # 7️⃣ Perform Detailed Requirements Analysis & Generate PDF
    print("\n[INFO] Running Detailed Requirements Analysis & Compliance Mapping...")
    try:
        from requirements_analysis import analyze_requirement
        from pdf_generator import generate_pdf_report

        analysis = analyze_requirement(cleaned)
        pdf_path = "output/requirements_analysis.pdf"
        generate_pdf_report(analysis, pdf_path)
        print(f"[SUCCESS] Requirements Analysis PDF successfully generated: {pdf_path}")
        print(f"          - MoSCoW Priority: {analysis.get('moscow_priority')}")
        print(f"          - Risk Score: {analysis.get('risk_level')} ({analysis.get('risk_score_pct')}/100)")
        print(f"          - Regulations Mapped: {', '.join(r.get('regulation') for r in analysis.get('regulations', []))}")
    except Exception as exc:
        print(f"[WARN] Could not generate PDF report: {exc}")


if __name__ == "__main__":
    main()
