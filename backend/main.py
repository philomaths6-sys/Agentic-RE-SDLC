# backend/main.py
"""FastAPI backend for Agentic RE‑SDLC system with fine-tuned models & RAG.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Path, Body, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="Agentic RE‑SDLC Backend")

# In‑memory stores
SESSIONS: Dict[str, Dict[str, Any]] = {}
REQUIREMENTS: Dict[str, Dict[str, Any]] = {}
SDLC_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {}
APPROVALS: List[Dict[str, Any]] = []
ARTEFACTS: Dict[str, Dict[str, Any]] = {}


class CreateSessionResponse(BaseModel):
    session_id: str


class MessageRequest(BaseModel):
    content: str


class ApprovalRequest(BaseModel):
    action: str  # "approve", "reject", "modify"
    comment: Optional[str] = None


@app.post("/session", response_model=CreateSessionResponse)
def create_session():
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"messages": [], "uploaded_files": []}
    return {"session_id": session_id}


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

    # Trigger RAG auto-ingestion
    try:
        from rag.ingest import main as ingest_main
        ingest_main()
    except Exception as exc:
        print(f"[WARN] Auto-ingestion warning: {exc}")

    return {
        "status": "success",
        "filename": filename,
        "filepath": file_path,
        "message": "Document saved to data/indian_docx and ingested into RAG store."
    }


@app.post("/session/{session_id}/message")
def post_message(
    session_id: str = Path(..., description="Session identifier"),
    payload: MessageRequest = Body(...),
):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    user_text = payload.content
    SESSIONS[session_id]["messages"].append({"role": "user", "content": user_text})

    # 1. Detailed Requirements Analysis & Compliance Mapping
    analysis_data = {}
    pdf_path = "output/requirements_analysis.pdf"
    try:
        from requirements_analysis import analyze_requirement
        from pdf_generator import generate_pdf_report

        analysis_data = analyze_requirement(user_text)
        generate_pdf_report(analysis_data, pdf_path)
        REQUIREMENTS[session_id] = {
            "status": "pending_approval",
            "requirement": user_text,
            "analysis": analysis_data,
            "pdf_path": pdf_path
        }
    except Exception as exc:
        print(f"[ERROR] Requirements analysis failed: {exc}")
        analysis_data = {"error": str(exc), "requirement_statement": user_text}
        REQUIREMENTS[session_id] = {
            "status": "error",
            "requirement": user_text,
            "analysis": analysis_data
        }

    mp = str(analysis_data.get('moscow_priority', 'N/A'))
    rl = str(analysis_data.get('risk_level', 'N/A'))
    response_text = f"[Extraction Agent] Extracted requirements for Indian banking domain.\n- MoSCoW Priority: {mp}\n- Risk Level: {rl}\n- PDF Generated: {pdf_path}"

    SESSIONS[session_id]["messages"].append({"role": "assistant", "content": response_text})

    return {
        "response": response_text,
        "requirements": REQUIREMENTS[session_id]
    }


@app.get("/requirements/{session_id}")
def get_requirements(session_id: str = Path(...)):
    if session_id not in REQUIREMENTS:
        raise HTTPException(status_code=404, detail="No requirements found for session")
    return REQUIREMENTS[session_id]


@app.post("/sdlc/{session_id}/recommend")
def generate_sdlc_recommendation(session_id: str = Path(...)):
    if session_id not in REQUIREMENTS:
        raise HTTPException(status_code=400, detail="Requirements must be created before SDLC prediction")

    req_data = REQUIREMENTS[session_id]
    statement = req_data.get("requirement", "")

    # 2. Invoke Fine-Tuned SDLC Recommendation Agent
    try:
        from agents.sdlc_agent import recommend_sdlc_structured
        sdlc_data = recommend_sdlc_structured(statement)
        sdlc_result = {
            "status": "pending_approval",
            "recommended_model": sdlc_data.get("recommended_model", "Agile-DevSecOps Hybrid"),
            "reasoning": sdlc_data.get("reasoning", ""),
            "alternatives_considered": sdlc_data.get("alternatives_considered", []),
        }
        SDLC_RECOMMENDATIONS[session_id] = sdlc_result
    except Exception as exc:
        print(f"[ERROR] SDLC recommendation failed: {exc}")
        sdlc_result = {
            "status": "error",
            "recommended_model": "",
            "reasoning": str(exc),
            "alternatives_considered": []
        }
        SDLC_RECOMMENDATIONS[session_id] = sdlc_result

    return SDLC_RECOMMENDATIONS[session_id]


@app.get("/sdlc/{session_id}")
def get_sdlc(session_id: str = Path(...)):
    if session_id not in SDLC_RECOMMENDATIONS:
        raise HTTPException(status_code=404, detail="No SDLC recommendation for this session")
    return SDLC_RECOMMENDATIONS[session_id]


@app.post("/approve/{session_id}/{item_type}")
def approve_item(
    session_id: str = Path(...),
    item_type: str = Path(..., description="'requirements' or 'sdlc'"),
    payload: ApprovalRequest = Body(...)
):
    record = {
        "session_id": session_id,
        "item_type": item_type,
        "action": payload.action,
        "comment": payload.comment
    }
    APPROVALS.append(record)

    if item_type == "requirements" and session_id in REQUIREMENTS:
        REQUIREMENTS[session_id]["status"] = payload.action
    elif item_type == "sdlc" and session_id in SDLC_RECOMMENDATIONS:
        SDLC_RECOMMENDATIONS[session_id]["status"] = payload.action

    return {"status": "recorded", "approval": record}


@app.get("/health")
def health_check():
    return {"status": "ok"}
