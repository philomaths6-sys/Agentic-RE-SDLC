# synthesis/__init__.py
"""Synthesis Layer for Requirements Engineering & Artefact Compilation."""

from synthesis.requirement_assembler import (
    ARCHITECTURAL_PILLARS,
    assemble_modular_requirements,
    synthesize_regulations,
    synthesize_technical_controls,
    build_traceability_matrix,
)

__all__ = [
    "ARCHITECTURAL_PILLARS",
    "assemble_modular_requirements",
    "synthesize_regulations",
    "synthesize_technical_controls",
    "build_traceability_matrix",
]
