# agents/requirement_generator.py
"""Utility to generate a requirement from a user query, applying guardrails.
Used by the main agent and the SDLC service.
"""
import os
import sys
import torch
from typing import List

# Guardrails
from guardrails.input_filter import is_safe as input_is_safe
from guardrails.output_filter import filter_output

# Retrieval
from rag.retriever import retrieve

# LLM loading (same as agents/main)
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
    context_texts = [c.get("text", "") for c in contexts]
    context_block = "\n---\n".join(context_texts)
    prompt = (
        "You are an expert requirements engineer for financial services. "
        "Using the following regulatory context, draft a precise requirement that addresses the user's request.\n\n"
        f"Context:\n{context_block}\n\n"
        f"User request: {query}\n\n"
        "Requirement:"
    )
    return prompt

def generate_requirement(user_query: str) -> str:
    """Generate a requirement string for the given user query.
    Returns the cleaned requirement text or raises ``ValueError`` on guard failure.
    """
    if not input_is_safe(user_query):
        raise ValueError("Input rejected by guardrails")
    retrieved = retrieve(user_query, top_k=5)
    prompt = _build_prompt(user_query, retrieved)
    llm = _load_llm()
    generated = llm(prompt)[0]["generated_text"]
    allowed, cleaned, reason = filter_output(generated)
    if not allowed:
        raise ValueError(f"Output rejected by guardrails: {reason}")
    return cleaned
