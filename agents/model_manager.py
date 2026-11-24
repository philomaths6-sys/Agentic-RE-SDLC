# agents/model_manager.py
"""Centralized Model Manager for Local 4-bit Qwen-2.5-3B-Instruct.
Loads the base model ONCE into GPU memory (~2.08 GB VRAM) to prevent OOM.
Supports hot-swapping and attaching both Extraction and SDLC LoRA adapters on demand using PEFT set_adapter().
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
EXTRACTION_ADAPTER_PATH = "finetune/adapters/extraction_adapter"
SDLC_ADAPTER_PATH = "finetune/adapters/sdlc_adapter"

_tokenizer = None
_base_model = None
_peft_model = None
_active_adapter = None


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


def get_peft_model():
    """Load base model once and attach both Extraction and SDLC LoRA adapters."""
    global _peft_model, _base_model, _tokenizer
    if _peft_model is not None:
        return _tokenizer, _peft_model

    tokenizer, base = get_base_model()

    has_ext = os.path.exists(EXTRACTION_ADAPTER_PATH)
    has_sdlc = os.path.exists(SDLC_ADAPTER_PATH)

    if has_ext:
        print(f"[INFO] Attaching fine-tuned Extraction LoRA adapter from: {EXTRACTION_ADAPTER_PATH}")
        _peft_model = PeftModel.from_pretrained(base, EXTRACTION_ADAPTER_PATH, adapter_name="extraction")
        if has_sdlc:
            print(f"[INFO] Loading fine-tuned SDLC LoRA adapter from: {SDLC_ADAPTER_PATH}")
            _peft_model.load_adapter(SDLC_ADAPTER_PATH, adapter_name="sdlc")
        _peft_model.eval()
        print(f"[INFO] Multi-adapter PeftModel ready. Total VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        return tokenizer, _peft_model
    elif has_sdlc:
        print(f"[INFO] Attaching fine-tuned SDLC LoRA adapter from: {SDLC_ADAPTER_PATH}")
        _peft_model = PeftModel.from_pretrained(base, SDLC_ADAPTER_PATH, adapter_name="sdlc")
        _peft_model.eval()
        return tokenizer, _peft_model
    else:
        return tokenizer, base


def get_extraction_model():
    """Return tokenizer and model with Extraction LoRA adapter set active."""
    global _active_adapter
    tokenizer, model = get_peft_model()
    if hasattr(model, "set_adapter") and "extraction" in getattr(model, "peft_config", {}):
        if _active_adapter != "extraction":
            model.set_adapter("extraction")
            _active_adapter = "extraction"
            print("[INFO] Activated active PEFT adapter: 'extraction'")
    return tokenizer, model


def get_sdlc_model():
    """Return tokenizer and model with SDLC LoRA adapter set active."""
    global _active_adapter
    tokenizer, model = get_peft_model()
    if hasattr(model, "set_adapter") and "sdlc" in getattr(model, "peft_config", {}):
        if _active_adapter != "sdlc":
            model.set_adapter("sdlc")
            _active_adapter = "sdlc"
            print("[INFO] Activated active PEFT adapter: 'sdlc'")
    return tokenizer, model
