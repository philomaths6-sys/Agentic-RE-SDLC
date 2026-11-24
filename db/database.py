# db/database.py
"""PostgreSQL Database Layer with SQLAlchemy ORM.
Handles:
- User Authentication (email + password hashing)
- Persistent Chat Sessions & Messages
- Persistent Requirements & SDLC Specifications
- Audit Trails
"""

import os
import uuid
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Boolean,
    DateTime, ForeignKey
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

load_dotenv()

# Build Connection URL from .env
PG_USER = os.getenv("POSTGRES_USER", "agentic_user")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "AgenticPass123!")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5433")
PG_DB = os.getenv("POSTGRES_DB", "agentic_re_sdlc")

DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Password Security (Salted PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, key_hex = stored_hash.split("$", 1)
        test_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return secrets.compare_digest(test_key.hex(), key_hex)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SQLAlchemy Models
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False, default="")
    role = Column(String(100), nullable=False, default="Principal Architect")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")
    state = relationship("SessionStateRecord", back_populates="session", uselist=False, cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")


class SessionStateRecord(Base):
    __tablename__ = "session_state_records"

    session_id = Column(String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE"), primary_key=True)
    analysis_json = Column(JSONB, nullable=True)
    sdlc_json = Column(JSONB, nullable=True)
    requirements_approved = Column(Boolean, default=False)
    sdlc_approved = Column(Boolean, default=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="state")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    actor_email = Column(String(255), nullable=False)
    action = Column(String(100), nullable=False, index=True)
    session_id = Column(String(64), nullable=True, index=True)
    payload = Column(JSONB, nullable=True)
    requirement_version = Column(String(50), nullable=True, default="1.0")
    model_version = Column(String(100), nullable=True, default="Qwen-2.5-3B-LoRA")
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class GateDecision(Base):
    __tablename__ = "gate_decisions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), nullable=True, index=True)
    gate_name = Column(String(50), nullable=False)
    decision = Column(String(20), nullable=False)
    reviewer_id = Column(Integer, nullable=True)
    reviewer_email = Column(String(255), nullable=False)
    reviewer_role = Column(String(100), nullable=False)
    rationale = Column(Text, nullable=True)
    model_name = Column(String(100), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


# ---------------------------------------------------------------------------
# Initialization & Seed
# ---------------------------------------------------------------------------
def init_db():
    """Create all tables and seed default demo user if empty."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        admin = db.query(User).filter(User.email == "architect@bank.in").first()
        if not admin:
            default_user = User(
                email="architect@bank.in",
                password_hash=hash_password("BankPass123!"),
                full_name="Lead FinTech Architect",
                role="Principal Architect"
            )
            db.add(default_user)
            db.commit()


# ---------------------------------------------------------------------------
# User Authentication API
# ---------------------------------------------------------------------------
def register_user(email: str, password: str, full_name: str = "", role: str = "Principal Architect") -> Dict[str, Any]:
    email_clean = email.strip().lower()
    if not email_clean or "@" not in email_clean:
        return {"success": False, "error": "Invalid email address format."}
    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}

    with SessionLocal() as db:
        existing = db.query(User).filter(User.email == email_clean).first()
        if existing:
            return {"success": False, "error": "An account with this email already exists."}

        new_user = User(
            email=email_clean,
            password_hash=hash_password(password),
            full_name=full_name.strip() or email_clean.split("@")[0].title(),
            role=role
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {
            "success": True,
            "user": {
                "id": new_user.id,
                "email": new_user.email,
                "full_name": new_user.full_name,
                "role": new_user.role
            }
        }


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    email_clean = email.strip().lower()
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email_clean).first()
        if not user or not verify_password(password, user.password_hash):
            return None
        return {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        }


# ---------------------------------------------------------------------------
# Session & Chat Persistence API
# ---------------------------------------------------------------------------
def get_user_chat_sessions(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve all chat sessions for the user ordered by recent activity."""
    with SessionLocal() as db:
        sessions = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(25)
            .all()
        )
        result = []
        for s in sessions:
            result.append({
                "id": s.id,
                "title": s.title,
                "updated_at": s.updated_at.isoformat() if s.updated_at else ""
            })
        return result


def load_chat_session(session_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Load complete session details including messages, analysis, and SDLC approval."""
    with SessionLocal() as db:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .first()
        )
        if not session:
            return None

        messages = [
            {"role": m.role, "content": m.content}
            for m in session.messages
        ]

        state = session.state
        return {
            "session_id": session.id,
            "title": session.title,
            "messages": messages,
            "analysis": state.analysis_json if state else None,
            "sdlc": state.sdlc_json if state else None,
            "req_approved": state.requirements_approved if state else False,
            "sdlc_approved": state.sdlc_approved if state else False
        }


def save_chat_session_state(
    user_id: int,
    session_id: str,
    title: str,
    messages: List[Dict[str, str]],
    analysis: Optional[Dict[str, Any]] = None,
    sdlc: Optional[Dict[str, Any]] = None,
    req_approved: bool = False,
    sdlc_approved: bool = False
) -> bool:
    """Upsert session, chat messages, and workflow state in PostgreSQL."""
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            session = ChatSession(
                id=session_id,
                user_id=user_id,
                title=title[:250],
                created_at=now,
                updated_at=now
            )
            db.add(session)
            db.flush()
        else:
            session.title = title[:250]
            session.updated_at = now

        # Synchronize messages
        db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        for msg in messages:
            db.add(ChatMessage(
                session_id=session_id,
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                created_at=now
            ))

        # Synchronize state record
        state = db.query(SessionStateRecord).filter(SessionStateRecord.session_id == session_id).first()
        if not state:
            state = SessionStateRecord(
                session_id=session_id,
                analysis_json=analysis,
                sdlc_json=sdlc,
                requirements_approved=req_approved,
                sdlc_approved=sdlc_approved,
                updated_at=now
            )
            db.add(state)
        else:
            state.analysis_json = analysis
            state.sdlc_json = sdlc
            state.requirements_approved = req_approved
            state.sdlc_approved = sdlc_approved
            state.updated_at = now

        db.commit()
        return True


def delete_chat_session(session_id: str, user_id: int) -> bool:
    with SessionLocal() as db:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .first()
        )
        if session:
            db.delete(session)
            db.commit()
            return True
        return False


# ---------------------------------------------------------------------------
# Append-Only Audit Trail & Stage-Gate Decision Governance API
# ---------------------------------------------------------------------------
def record_audit_log(
    actor_email: str,
    action: str,
    payload: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[int] = None,
    requirement_version: str = "1.0",
    model_version: str = "Qwen-2.5-3B-LoRA"
) -> Dict[str, Any]:
    """Append-only immutable audit trail record."""
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        log_entry = AuditLog(
            user_id=user_id,
            actor_email=actor_email,
            action=action,
            session_id=session_id,
            payload=payload or {},
            requirement_version=requirement_version,
            model_version=model_version,
            timestamp=now
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        return {
            "id": log_entry.id,
            "actor_email": log_entry.actor_email,
            "action": log_entry.action,
            "session_id": log_entry.session_id,
            "requirement_version": log_entry.requirement_version,
            "model_version": log_entry.model_version,
            "timestamp": log_entry.timestamp.isoformat()
        }


def record_gate_decision(
    session_id: str,
    gate_name: str,
    decision: str,
    reviewer_email: str,
    reviewer_role: str,
    reviewer_id: Optional[int] = None,
    rationale: str = "",
    model_name: str = ""
) -> Dict[str, Any]:
    """Record a formal stage-gate decision with verified reviewer identity."""
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        gate_entry = GateDecision(
            session_id=session_id,
            gate_name=gate_name,
            decision=decision,
            reviewer_id=reviewer_id,
            reviewer_email=reviewer_email,
            reviewer_role=reviewer_role,
            rationale=rationale,
            model_name=model_name,
            timestamp=now
        )
        db.add(gate_entry)

        # Also write to append-only audit_log
        audit_entry = AuditLog(
            user_id=reviewer_id,
            actor_email=reviewer_email,
            action=f"STAGE_GATE_{gate_name}_{decision}",
            session_id=session_id,
            payload={
                "gate_name": gate_name,
                "decision": decision,
                "reviewer_email": reviewer_email,
                "reviewer_role": reviewer_role,
                "rationale": rationale,
                "model_name": model_name
            },
            requirement_version="1.0",
            model_version="Qwen-2.5-3B-LoRA",
            timestamp=now
        )
        db.add(audit_entry)
        db.commit()
        db.refresh(gate_entry)
        return {
            "id": gate_entry.id,
            "session_id": gate_entry.session_id,
            "gate_name": gate_entry.gate_name,
            "decision": gate_entry.decision,
            "reviewer_email": gate_entry.reviewer_email,
            "reviewer_role": gate_entry.reviewer_role,
            "rationale": gate_entry.rationale,
            "timestamp": gate_entry.timestamp.isoformat()
        }


def get_audit_logs(session_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Query immutable audit log entries."""
    with SessionLocal() as db:
        q = db.query(AuditLog)
        if session_id:
            q = q.filter(AuditLog.session_id == session_id)
        entries = q.order_by(AuditLog.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": e.id,
                "actor_email": e.actor_email,
                "action": e.action,
                "session_id": e.session_id,
                "payload": e.payload,
                "requirement_version": e.requirement_version,
                "model_version": e.model_version,
                "timestamp": e.timestamp.isoformat() if e.timestamp else ""
            }
            for e in entries
        ]


def get_gate_decisions(session_id: str) -> List[Dict[str, Any]]:
    """Retrieve all gate decisions for a session."""
    with SessionLocal() as db:
        decisions = (
            db.query(GateDecision)
            .filter(GateDecision.session_id == session_id)
            .order_by(GateDecision.timestamp.asc())
            .all()
        )
        return [
            {
                "id": d.id,
                "gate_name": d.gate_name,
                "decision": d.decision,
                "reviewer_email": d.reviewer_email,
                "reviewer_role": d.reviewer_role,
                "rationale": d.rationale,
                "model_name": d.model_name,
                "timestamp": d.timestamp.isoformat() if d.timestamp else ""
            }
            for d in decisions
        ]
