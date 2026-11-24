# telemetry/pipeline_tracer.py
"""Pipeline Execution Telemetry & Stage Tracking Engine.
Tracks real execution timing, stage transitions, latency, and status metrics
across Zero-Trust Guardrails, RAG Grounding, LLM Model Extraction, and Verification.
"""

import time
from typing import List, Dict, Any, Optional, Callable


class PipelineTracer:
    """Dynamic telemetry tracker for pipeline stage execution."""

    def __init__(self, step_callback: Optional[Callable[[str, str], None]] = None):
        self.step_callback = step_callback
        self.stages: List[Dict[str, Any]] = []
        self._current_stage_start: Optional[float] = None

    def log_event(self, stage_name: str, message: str):
        """Notify real-time callback subscriber (e.g. Streamlit UI status box)."""
        if self.step_callback:
            try:
                self.step_callback(stage_name, message)
            except Exception:
                pass

    def record_stage(
        self,
        stage_id: str,
        stage_name: str,
        role: str,
        status: str,
        badge: str,
        badge_color: str,
        output_summary: str,
        details: Dict[str, Any],
        duration_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """Record a completed or evaluated stage in the execution trace."""
        record = {
            "agent_id": stage_id,
            "agent_name": stage_name,
            "stage_id": stage_id,
            "stage_name": stage_name,
            "role": role,
            "status": status,
            "badge": badge,
            "badge_color": badge_color,
            "output_summary": output_summary,
            "duration_ms": duration_ms or 0.0,
            "details": details
        }
        self.stages.append(record)
        return record

    def get_trace(self) -> List[Dict[str, Any]]:
        """Return the complete sequence of executed pipeline stages."""
        return list(self.stages)


def build_conversational_trace() -> List[Dict[str, Any]]:
    """Build standardized trace for conversational intent handling."""
    return [
        {
            "agent_id": "guardrail_perimeter",
            "agent_name": "Perimeter Guardrail",
            "stage_id": "guardrail_perimeter",
            "stage_name": "Perimeter Guardrail",
            "role": "Zero-Trust Perimeter Gatekeeper",
            "status": "PASSED",
            "badge": "Input Gate",
            "badge_color": "#A78BFA",
            "output_summary": "Classified input as conversational interaction. Routed to interactive conversational handler.",
            "details": {
                "guardrail_status": "Verified Safe",
                "intent": "Conversational Greeting / Inquiry",
                "routing": "Conversational Response Handler"
            }
        }
    ]


def build_security_rejection_trace(reason_msg: str, threat_assessment: str) -> List[Dict[str, Any]]:
    """Build standardized trace for security guardrail intercept."""
    return [
        {
            "agent_id": "guardrail_perimeter",
            "agent_name": "Perimeter Guardrail",
            "stage_id": "guardrail_perimeter",
            "stage_name": "Perimeter Guardrail",
            "role": "Zero-Trust Security Perimeter",
            "status": "REJECTED (Security Guardrail)",
            "badge": "Security Alert",
            "badge_color": "#EF4444",
            "output_summary": f"Input blocked by Zero-Trust security filter ({reason_msg}).",
            "details": {
                "guardrail_status": "Blocked (Security Violation)",
                "threat_assessment": threat_assessment,
                "action": "Execution halted to prevent adversarial injection."
            }
        }
    ]


def build_domain_rejection_trace(detected_category: str, threat_assessment: str) -> List[Dict[str, Any]]:
    """Build standardized trace for out-of-domain scope intercept."""
    return [
        {
            "agent_id": "domain_classifier",
            "agent_name": "Domain Scope Classifier",
            "stage_id": "domain_classifier",
            "stage_name": "Domain Scope Classifier",
            "role": "Affirmative Financial Domain Scope Gate",
            "status": "REJECTED (Domain Boundary)",
            "badge": "Scope Alert",
            "badge_color": "#F59E0B",
            "output_summary": f"Query classified as out-of-scope ({detected_category}). Intercepted at supervisor gate to prevent model hallucination.",
            "details": {
                "guardrail_status": "Out of Domain",
                "threat_assessment": threat_assessment,
                "detected_topic": detected_category,
                "resolution": "Returned domain boundary clarification to user."
            }
        }
    ]
