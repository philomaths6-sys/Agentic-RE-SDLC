# requirements_analysis.py
"""Unified Entry Point & Facade for Indian Banking & Financial Requirements Engineering.

Delegates execution to the LangGraph stateful multi-agent system (`workflow.graph_orchestrator`),
re-exporting canonical domain models, risk engines, and guardrails to ensure 100% backward compatibility
across CLI benchmarks, Streamlit UI, and FastAPI backend while strictly eliminating code duplication (DRY).
"""

from typing import Dict, Any, Optional

# Re-export canonical domain catalog & detection from single source of truth
from workflow.graph_orchestrator import (
    run_financial_engineering_graph,
    _detect_financial_domain,
    FINANCIAL_DOMAIN_CATALOG,
    get_domain_display_name,
    get_domain_category,
    get_domain_statutory,
)

# Re-export core functional utilities for benchmark & script backward compatibility
from governance.risk_engine import calculate_dynamic_risk
from guardrails.input_filter import is_conversational


def analyze_requirement(requirement_text: str, step_callback: Optional[Any] = None) -> Dict[str, Any]:
    """Execute end-to-end financial engineering pipeline using LangGraph stateful multi-agent system.
    
    Includes:
      - Zero-Trust Input Guardrails & Perimeter Filter
      - Qdrant Vector DB Regulatory RAG Grounding
      - Fine-Tuned 4-bit LoRA Extraction Agent
      - Modular Specification Assembler (IEEE 830)
      - Verifier & Critic Agent (Ambiguity, Testability, Duplicates, Contradictions)
      - Dynamic Revision & Targeted Regeneration Reflection Loop
      - Hybrid Multi-Factor Regulatory Risk Engine
      - Full Per-Stage Input-Output Observability Logging
    """
    if not requirement_text or not requirement_text.strip():
        raise ValueError("Empty requirement text provided.")

    return run_financial_engineering_graph(requirement_text, step_callback=step_callback)
