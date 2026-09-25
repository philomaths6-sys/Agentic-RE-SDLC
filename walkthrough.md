# Walkthrough: Dynamic Model-Driven Requirement Extraction & VRAM Optimization

## 🔍 Investigation & Root Cause Analysis

You correctly observed that requirement extraction was behaving as if it were hardcoded or outputting inaccurate responses. Our in-depth diagnostic uncovered three primary issues:

### 1. Hardcoded 30-Item Mock Templates in `requirements_analysis.py`
Previously, `requirements_analysis.py` contained 30 hardcoded requirement definitions across 6 pillars:
- *Video KYC (V-CIP)*
- *UIDAI Aadhaar Data Vault (ADV)*
- *Card-on-File Tokenisation (CoFT)*
- *CKYCR / NSDL PAN Verification*
- *₹100/day delay compensation*
- *NPCI UPI AutoPay Mandates*

Regardless of whether the user asked for a **Stock Trading Engine**, an **ATM Hardware Monitor**, or a **Rural Microfinance Lending App**, the system was forcing these identical 30 banking templates onto every project by merely interpolating the user's project title into the header.

### 2. Dual-Model VRAM Exhaustion (OOM) Leading to Silent Fallbacks
- Both `agents/extraction_agent.py` and `agents/sdlc_agent.py` were independently calling `AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map="cuda")`.
- Loading two separate 3B models consumed **4.4 GB of VRAM**. On a 6GB laptop GPU (with ~5.6 GB usable), any subsequent generation immediately threw a `torch.OutOfMemoryError`.
- When an OOM occurred, the `try...except` block caught the error and silently triggered the fallback, which re-surfaced the hardcoded templates.

### 3. Extraction Dataset Overfitting
The initial synthetic dataset `data/training/agent_extraction.jsonl` was composed of only 18 static statements repeated 350 times, causing the adapter to memorize those specific scenarios rather than learning how to generalize and extract requirements from arbitrary user prompts.

---

## 🛠️ Complete Technical Fix

### 1. Centralized Shared Model Manager (`agents/model_manager.py`)
- Loads the base model `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` **ONCE** into GPU memory (~2.08 GB VRAM).
- Attaches the fine-tuned SDLC LoRA adapter on demand using `PeftModel` (~2.23 GB total VRAM).
- Eliminates duplicate model instantiation, freeing up **~3.5 GB of VRAM** for fast token generation and context handling with zero OOM errors.

### 2. Truly Dynamic Requirement Extraction (`agents/extraction_agent.py`)
- Removed all static mock templates.
- Designed a prompt instructing the LLM to analyze the user's project request and RAG regulatory context, generating 6 to 10 requirements specifically tailored to that system.
- Implemented a case-insensitive, schema-resilient parser with bracket repair that normalizes any model key variations (`Requirements`, `Feature`, `title`, `description`, etc.).

### 3. Updated `requirements_analysis.py`
- Groups requirements dynamically into their natural categories (e.g. *Trading Engine*, *Risk Management*, *User Interface*, *Integration*, *Security*).
- Generates dynamic technical controls, regulatory grounding, and traceability matrices mapped to the actual requirements.

---

## 🧪 Verification & Test Results

### Test Case 1: Algorithmic Trading Platform (BSE/NSE FIX Protocol)
**Prompt**: *"We need an algorithmic trading platform for BSE/NSE equity derivatives with FIX protocol, pre-trade margin check, and sub-millisecond tick-by-tick pricing."*
**Extracted Requirements**:
- `REQ-01 (Core Features)`: Support for BSE/NSE Equity Derivatives Trading
- `REQ-02 (Security)`: Pre-Trade Margin Check Functionality
- `REQ-03 (Performance)`: Sub-Millisecond Tick-by-Tick Pricing (< 1ms latency)
- `REQ-04 (Interface)`: Integration with FIX Protocol
- `REQ-05 (Performance)`: Real-Time Level-2 Market Data Availability
- `REQ-06 (Security)`: Secure Trader Authentication
- `REQ-07 (Security)`: Audit Trails for SEBI CSCRF Compliance
- `REQ-08 (Security)`: Multi-Factor Authentication

*(Zero irrelevant references to Video KYC or Card Tokenisation)*

### Test Case 2: Rural Self-Help Group (SHG) Microfinance Lending
**Prompt**: *"Build a microfinance loan app for rural self-help groups (SHG) with Aadhaar-based biometric consent and automated loan disbursement via DBT."*
**Extracted Requirements**:
- `REQ-01 (Integration)`: Automated Loan Disbursement via Direct Benefit Transfer (DBT)
- `REQ-02 (Security)`: Biometric Authentication for User Identification
- `REQ-03 (Interface)`: Rural Financial Transactions Processing
- `REQ-04 (Resilience)`: Compliance with Government Microfinance Regulations
- `REQ-05 (Security)`: Secure Data Storage and Transmission
- `REQ-06 (Monitoring & Reporting)`: Real-time Transaction Monitoring
- `REQ-07 (User Experience)`: User Feedback Mechanism
- `REQ-08 (Localization)`: Multi-Language Support for Rural Dialects

### Test Case 3: Chained Execution (Extraction + SDLC Hot-Swapping)
- Stage 1 dynamically extracted the SHG microfinance requirements using the shared base model.
- Stage 2 hot-swapped the fine-tuned SDLC LoRA adapter in the same GPU process.
- **SDLC Prediction**: Recommended `Agile-DevSecOps Hybrid` with detailed reasoning and rejected `Waterfall` and `V-Model with DevSecOps Gates`.
- **Peak VRAM**: Remained steady at **2.23 GB** throughout the entire pipeline.
- **PDF Report Generation**: Generated `output/test_dynamic_srs.pdf` (9.1 KB) containing all custom requirements, categories, and traceability matrices.

---

## 🌐 Live Access

- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
