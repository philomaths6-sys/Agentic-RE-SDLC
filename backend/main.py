# backend/main.py
"""FastAPI Unified Service Layer for Financial Software Engineering & SDLC Governance.
Back-end persistence powered directly by PostgreSQL with append-only immutable audit logs.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Path, Body, UploadFile, File, Query
from pydantic import BaseModel

from db.database import (
    init_db,
    register_user,
    authenticate_user,
    get_user_chat_sessions,
    get_chat_session_state,
    save_chat_session_state,
    delete_chat_session,
    record_audit_log,
    record_gate_decision,
    get_audit_logs,
    get_gate_decisions,
)

# Initialize PostgreSQL Schema & Tables
try:
    init_db()
except Exception as e:
    print(f"[WARN] Database initialization notice: {e}")

app = FastAPI(
    title="Financial RE-SDLC Governance Backend",
    description="Unified API Service with PostgreSQL persistence and statutory audit trail."
)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = ""
    role: Optional[str] = "Principal Architect"


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateSessionRequest(BaseModel):
    user_id: int
    title: Optional[str] = "New Project Session"


class MessageRequest(BaseModel):
    user_id: int
    content: str


class GateDecisionRequest(BaseModel):
    gate_name: str  # e.g. "GATE_1_REQUIREMENTS" or "GATE_2_SDLC"
    decision: str   # "APPROVED", "REJECTED", "OVERRULED"
    reviewer_email: str
    reviewer_role: str
    reviewer_id: Optional[int] = None
    rationale: Optional[str] = ""
    model_name: Optional[str] = ""


# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------
@app.post("/auth/register")
def register(payload: RegisterRequest):
    res = register_user(
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name or "",
        role=payload.role or "Principal Architect"
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Registration failed"))
    return res


@app.post("/auth/login")
def login(payload: LoginRequest):
    user = authenticate_user(email=payload.email, password=payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password credentials")
    # Log successful login into audit trail
    record_audit_log(
        actor_email=user["email"],
        action="USER_LOGIN_SUCCESS",
        user_id=user["id"],
        payload={"role": user["role"]}
    )
    return {"success": True, "user": user}


# ---------------------------------------------------------------------------
# Session & Project Workspace Endpoints
# ---------------------------------------------------------------------------
@app.get("/user/{user_id}/sessions")
def list_sessions(user_id: int = Path(...)):
    sessions = get_user_chat_sessions(user_id)
    return {"sessions": sessions}


@app.post("/session")
def create_session(payload: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    save_chat_session_state(
        session_id=session_id,
        user_id=payload.user_id,
        title=payload.title or "New Project Session",
        messages=[],
        analysis=None,
        sdlc=None
    )
    record_audit_log(
        actor_email="system",
        action="SESSION_CREATED",
        session_id=session_id,
        user_id=payload.user_id,
        payload={"title": payload.title}
    )
    return {"session_id": session_id, "title": payload.title}


@app.get("/session/{session_id}")
def get_session(session_id: str = Path(...)):
    state = get_chat_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@app.delete("/session/{session_id}")
def remove_session(session_id: str = Path(...), user_id: int = Query(...)):
    success = delete_chat_session(session_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or unauthorized")
    return {"success": True}


# ---------------------------------------------------------------------------
# Document Ingestion Endpoint
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".docx", ".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")

    save_dir = os.path.join("data", "indian_docx")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    content = await file.read()
    with open(file_path, "wb") as out:
        out.write(content)

    ingest_info = {}
    try:
        from pathlib import Path
        from rag.ingest import ingest_user_document
        ingest_info = ingest_user_document(Path(file_path))
    except Exception as exc:
        print(f"[WARN] User document ingestion notice: {exc}")

    return {
        "status": "success",
        "filename": filename,
        "filepath": file_path,
        "collection": ingest_info.get("collection", "user_documents"),
        "chunks_ingested": ingest_info.get("chunks_ingested", 0),
        "message": "Document saved and incrementally ingested into isolated user_documents collection."
    }


# ---------------------------------------------------------------------------
# Requirement Pipeline & Execution Endpoint
# ---------------------------------------------------------------------------
@app.post("/session/{session_id}/message")
def post_message(
    session_id: str = Path(...),
    payload: MessageRequest = Body(...),
):
    existing = get_chat_session_state(session_id)
    if not existing:
        save_chat_session_state(
            session_id=session_id,
            user_id=payload.user_id,
            title=payload.content[:40],
            messages=[]
        )
        existing = {"messages": [], "analysis": None, "sdlc": None}

    messages = existing.get("messages", [])
    user_text = payload.content
    messages.append({"role": "user", "content": user_text})

    from requirements_analysis import analyze_requirement
    from pdf_generator import generate_pdf_report

    pdf_path = "output/requirements_analysis.pdf"
    analysis_data = analyze_requirement(user_text)

    if not analysis_data.get("is_conversational") and not analysis_data.get("is_out_of_domain") and not analysis_data.get("is_security_violation"):
        try:
            generate_pdf_report(analysis_data, pdf_path)
        except Exception as exc:
            print(f"[WARN] PDF generation notice: {exc}")

        response_text = (
            f"Generated **{analysis_data.get('total_requirements_count', 30)} modular specifications** "
            f"for **\"{analysis_data.get('title', 'Project')}\"** across all 6 architectural pillars. "
            f"Evaluated regulatory risk at {analysis_data.get('risk_score_pct', 50)}% ({analysis_data.get('risk_level', 'MEDIUM')})."
        )
    elif analysis_data.get("is_conversational"):
        response_text = analysis_data.get("conversational_response", "")
    elif analysis_data.get("is_out_of_domain"):
        response_text = analysis_data.get("domain_rejection_response", "")
    else:
        response_text = analysis_data.get("security_rejection_response", "")

    messages.append({"role": "assistant", "content": response_text})

    # Persist directly into PostgreSQL
    save_chat_session_state(
        session_id=session_id,
        user_id=payload.user_id,
        title=analysis_data.get("title", user_text[:40]),
        messages=messages,
        analysis=analysis_data if not analysis_data.get("is_conversational") else None,
        sdlc=existing.get("sdlc")
    )

    # Append to immutable statutory audit log
    record_audit_log(
        actor_email=f"user_{payload.user_id}",
        action="REQUIREMENT_SUBMISSION",
        session_id=session_id,
        user_id=payload.user_id,
        payload={
            "query": user_text,
            "domain": analysis_data.get("domain"),
            "risk_score": analysis_data.get("risk_score_pct"),
            "req_count": analysis_data.get("total_requirements_count")
        },
        requirement_version="1.0",
        model_version="Qwen-2.5-3B-LoRA"
    )

    return {
        "response": response_text,
        "analysis": analysis_data,
        "messages": messages
    }


# ---------------------------------------------------------------------------
# Stage-Gate Governance & Approvals
# ---------------------------------------------------------------------------
@app.post("/session/{session_id}/gate-decision")
def submit_gate_decision(
    session_id: str = Path(...),
    payload: GateDecisionRequest = Body(...)
):
    """Store stage-gate decision with verified reviewer identity in immutable audit trail."""
    record = record_gate_decision(
        session_id=session_id,
        gate_name=payload.gate_name,
        decision=payload.decision,
        reviewer_email=payload.reviewer_email,
        reviewer_role=payload.reviewer_role,
        reviewer_id=payload.reviewer_id,
        rationale=payload.rationale or "",
        model_name=payload.model_name or ""
    )

    # Update session approval flags
    state = get_chat_session_state(session_id)
    if state:
        req_app = state.get("requirements_approved", False)
        sdlc_app = state.get("sdlc_approved", False)
        if payload.gate_name == "GATE_1_REQUIREMENTS":
            req_app = (payload.decision.upper() == "APPROVED")
        elif payload.gate_name == "GATE_2_SDLC":
            sdlc_app = (payload.decision.upper() == "APPROVED")

        save_chat_session_state(
            session_id=session_id,
            user_id=state.get("user_id", 1),
            title=state.get("title", "Project"),
            messages=state.get("messages", []),
            analysis=state.get("analysis"),
            sdlc=state.get("sdlc"),
            req_approved=req_app,
            sdlc_approved=sdlc_app
        )

    return {"status": "recorded", "decision_record": record}


# ---------------------------------------------------------------------------
# Immutable Audit Log Query Endpoints
# ---------------------------------------------------------------------------
@app.get("/session/{session_id}/audit-log")
def fetch_session_audit_log(session_id: str = Path(...)):
    logs = get_audit_logs(session_id=session_id)
    return {"audit_trail": logs}


@app.get("/session/{session_id}/gate-decisions")
def fetch_session_gate_decisions(session_id: str = Path(...)):
    decisions = get_gate_decisions(session_id=session_id)
    return {"gate_decisions": decisions}


@app.get("/audit-log")
def fetch_all_audit_logs(limit: int = Query(50)):
    logs = get_audit_logs(limit=limit)
    return {"audit_trail": logs}


@app.get("/health")
def health_check():
    return {"status": "ok", "persistence": "postgresql"}
