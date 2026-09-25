# agents/model_manager.py
"""Centralized Model Manager for Local 4-bit Qwen-2.5-3B-Instruct.
Loads the base model ONCE into GPU memory (~2.08 GB VRAM) to prevent OOM.
Supports hot-swapping and attaching LoRA adapters on demand.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
SDLC_ADAPTER_PATH = "finetune/adapters/sdlc_adapter"

_tokenizer = None
_base_model = None
_sdlc_model = None


def get_tokenizer():
    """Load and cache tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def get_base_model():
    """Load base model once and cache globally on GPU."""
    global _base_model, _tokenizer
    if _base_model is not None:
        return get_tokenizer(), _base_model

    print("[INFO] Loading shared base model Qwen-2.5-3B-Instruct into GPU...")
    _tokenizer = get_tokenizer()
    _base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="cuda",
    )
    _base_model.eval()
    print(f"[INFO] Shared base model loaded. VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return _tokenizer, _base_model


def get_sdlc_model():
    """Return tokenizer and model with SDLC LoRA adapter attached."""
    global _sdlc_model
    tokenizer, base = get_base_model()
    if _sdlc_model is not None:
        return tokenizer, _sdlc_model

    if os.path.exists(SDLC_ADAPTER_PATH):
        try:
            print(f"[INFO] Attaching fine-tuned SDLC LoRA adapter from: {SDLC_ADAPTER_PATH}")
            _sdlc_model = PeftModel.from_pretrained(base, SDLC_ADAPTER_PATH)
            _sdlc_model.eval()
            print(f"[INFO] SDLC adapter attached. Total VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            return tokenizer, _sdlc_model
        except Exception as exc:
            print(f"[WARN] Failed loading SDLC adapter ({exc}), falling back to base model.")
            return tokenizer, base
    else:
        return tokenizer, base
